using Statistics
using LinearAlgebra
using Random

using Distributions
using BandedMatrices

include("da.jl")
import .DA

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(1)

D = 40
model = Models.lorenz96
B = brand(40, 40, 20, 20)
Q_true = Matrix((B .- 0.4)*(B .- 0.4)')/10
model_err_prescribed = Q_true
model_true = Models.lorenz96

H = I(D)
ens_size = 80
model_size = D
integrator = Integrators.rk4
da_method = DA.etkf
x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(0=>0.4*ones(D)))
ens_errs = Symmetric(diagm(0=>0.4*ones(D)))
x0 = x[end, :]

n_cycles = 3000
ρ = 1e-3

save_Q_hist = true

window = 1

model_err = 0.1*diagm(0=>ones(D))

E = x0 .+ rand(MvNormal(ens_errs), ens_size)

info = ens_forecast.da_cycles(x0=x0, E=E, model=model, model_true=model_true,
                              H=H, model_err=model_err,
                              model_err_prescribed=model_err_prescribed,
                              integrator=integrator, da_method=da_method, Δt=Δt,
                              window=window, n_cycles=n_cycles, outfreq=outfreq,
                              model_size=model_size, R=R, ρ=ρ, Q_p=nothing,
                              save_Q_hist=save_Q_hist)