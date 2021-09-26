module ens_forecast

export da_cycles

using Statistics
using LinearAlgebra
using Random

using Distributions
using ProgressMeter
using PyCall

struct Forecast_Info
    errs
    errs_fcst
    crps
    crps_fcst
    spread
    spread_fcst
    Q_hist
    model_err
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function make_psd(A, tol=1e-6)
    L, Q = eigen(A)
    L[L .< 0] .= tol
    return Symmetric(Q*diagm(0=>L)*inv(Q))
end

function da_cycles(; x0::AbstractVector{float_type},
                     E::AbstractMatrix{float_type}, model::Function,
                     model_true::Function, H::AbstractMatrix,
                     model_err::AbstractMatrix{float_type},
                     model_err_prescribed::Union{AbstractMatrix{float_type}, Nothing}=nothing,
                     integrator::Function, da_method::Function, Δt::float_type,
                     window::int_type, n_cycles::int_type, outfreq::int_type,
                     model_size::int_type, R::Symmetric{float_type},
                     ρ::float_type, Q_p::Union{AbstractVector{<:AbstractMatrix{float_type}}, Nothing}=nothing,
                     save_Q_hist::Bool=false) where {float_type<:AbstractFloat, int_type<:Integer}
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    ens_size = size(E, 2)

    x_true = x0

    errs = Array{float_type}(undef, n_cycles, model_size)
    errs_fcst = Array{float_type}(undef, n_cycles, model_size)
    crps = Array{float_type}(undef, n_cycles)
    crps_fcst = Array{float_type}(undef, n_cycles)
    Q_hist = Array{float_type}(undef, n_cycles)
    if save_Q_hist
        Q_hist = Array{Matrix{float_type}}(undef, n_cycles)
    end
    spread = Array{float_type}(undef, n_cycles)
    spread_fcst = Array{float_type}(undef, n_cycles)

    t = 0.0

    @showprogress for cycle=1:n_cycles
        y = H*x_true + rand(obs_err_dist)

        x_m = mean(E, dims=2)
        innovation = y - H*x_m
        P_e = innovation*innovation'
        P_f = Symmetric(cov(E'))

        C = P_e - R - H*P_f*H'
        if rank(H) >= model_size
            Q_est = pinv(H)*C*pinv(H)'
        else
            A = Array{float_type}(undef, size(R)[1]^2, length(Q_p))
            for p=1:length(Q_p)
                A[:, p] = vec(H*Q_p[p]*H')
            end
            q = A \ vec(C)
            Q_est = sum([q[p]*Q_p[p] for p=1:length(Q_p)])
        end

        Q = Symmetric(ρ*Q_est + (1 - ρ)*model_err)

        if !isposdef(Q)
            Q = make_psd(Q)
        end

        if save_Q_hist
            Q_hist[cycle] = Q
        else
            Q_hist[cycle] = tr(Q)
        end
        model_err = Q

        E += rand(MvNormal(model_err), ens_size)

        errs_fcst[cycle, :] = mean(E, dims=2) - x_true

	    E_corr_fcst_array = xarray.DataArray(data=E, dims=["dim", "member"])
        crps_fcst[cycle] = xskillscore.crps_ensemble(x_true, E_corr_fcst_array).values[1]
        spread_fcst[cycle] = mean(std(E, dims=2))

        E = da_method(E=E, R=R, R_inv=R_inv, H=H, y=y)

        E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
        crps[cycle] = xskillscore.crps_ensemble(x_true, E_corr_array).values[1]

        spread[cycle] = mean(std(E, dims=2))

        errs[cycle, :] = mean(E, dims=2) - x_true

        if model_err_prescribed === nothing
            pert = zeros(model_size)
        else
            pert = rand(MvNormal(model_err_prescribed))
        end
        for i=1:ens_size
            integration = integrator(model, E[:, i], t,
                                     t + window*outfreq*Δt, Δt, inplace=false)
            E[:, i] = integration[end, :] + pert
        end

        x_true = integrator(model_true, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    return Forecast_Info(errs, errs_fcst, crps, crps_fcst, spread, spread_fcst,
                         Q_hist, model_err)
end

end
