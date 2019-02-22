include("nuts.jl"); using NUTSJulia;
using Gadfly

function f(x::Vector)
  d = ones(length(x)) * 3
  return exp(-(dot(x, x))) + exp(-(dot(x - d, x - d))) + exp(-(dot(x + d, x + d)))
end

θ0 = rand(2)
@time samples = naive_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
plot(x=[s[1] for s in samples], y=[s[2] for s in samples], Geom.point)


θ0 = rand(2)
@time samples = eff_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
plot(x=[s[1] for s in samples], y=[s[2] for s in samples], Geom.point)


θ0 = rand(2)
@time samples = NUTS(θ0, 0.65, x -> log(f(x)), 2000, 200)
plot(x=[s[1] for s in samples], y=[s[2] for s in samples], Geom.point)
