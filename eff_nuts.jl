# An Julia Implementation of Efficient No-U-Turn Sampler described in Algorithm 3 in Hoffman et al. (2011)
# Author: Kai Xu
# Date: 06/10/2016

function eff_NUTS(θ0, ϵ, L, M)
  doc"""
    - θ0      : initial model parameter
    - ϵ       : leapfrog step size
    - L       : likelihood function
    - M       : sample number
  """

  function leapfrog(θ, r, ϵ)
    doc"""
      - θ : model parameter
      - r : momentum variable
      - ϵ : leapfrog step size
    """
    r̃ = r + (ϵ / 2) * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + (ϵ / 2) * ∇L(θ̃)
    return θ̃, r̃
  end

  function build_tree(θ, r, u, v, j, ϵ)
    doc"""
      - θ   : model parameter
      - r   : momentum variable
      - u   : slice variable
      - v   : direction ∈ {-1, 1}
      - j   : depth
      - ϵ   : leapfrog step size
    """
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, v * ϵ)
      n′ = u <= exp(L(θ′) - 0.5 * dot(r′, r′))
      s′ = u < exp(Δ_max + L(θ′) - 0.5 * dot(r′, r′))
      return θ′, r′, θ′, r′, θ′, n′, s′
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′ = build_tree(θ, r, u, v, j - 1, ϵ)
      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′ = build_tree(θm, rm, u, v, j - 1, ϵ)
        else
          _, _, θp, rp, θ′′, n′′, s′′ = build_tree(θp, rp, u, v, j - 1, ϵ)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        s′ = s′′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
        n′ = n′ + n′′
      end
      return θm, rm, θp, rp, θ′, n′, s′
    end
  end

  ∇L = θ -> ForwardDiff.gradient(L, θ)  # generate gradient function

  θs = Array{Array}(M + 1)
  θs[1] = θ0

  println("[eff_NUTS] start sampling for $M samples with ϵ=$ϵ")

  for m = 1:M
    print('.')
    r0 = randn(length(θ0))
    u = rand() * exp(L(θs[m]) - 0.5 * dot(r0, r0)) # Note: θ^{m-1} in the paper corresponds to
                                                   #       `θs[m]` in the code
    θm, θp, rm, rp, j, θs[m + 1], n, s = θs[m], θs[m], r0, r0, 0, θs[m], 1, 1
    while s == 1
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, θ′, n′, s′ = build_tree(θm, rm, u, v_j, j, ϵ)
      else
        _, _, θp, rp, θ′, n′, s′ = build_tree(θp, rp, u, v_j, j, ϵ)
      end
      if s′ == 1
        if rand() < min(1, n′ / n)
          θs[m + 1] = θ′
        end
      end
      n = n + n′
      s = s′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
      j = j + 1
    end
  end

  println()
  println("[eff_NUTS] sampling complete")

  return θs
end
