module Models

function lorenz96(t, u)
   N = 40
   F = 8

   du = similar(u)

   for i=1:N
      du[i] = (u[mod(i+1, 1:N)] - u[mod(i-2, 1:N)])*u[mod(i-1, 1:N)] - u[i] + F
   end

   return copy(du)
end

end
