# Simple regulator (ADM Appendix F.5). Mirrors adm/problems/simple_regulator.py;
# tests/test_problems_parity.py checks that both languages give the same exact values.
# s′ ~ N(s + a, 0.1²), reward -s², start s ~ N(0.3, 0.1²); γ = 0.9 and 10 steps are this site's choice.
const REGULATOR_NOISE_SD = 0.1
const REGULATOR_START = (0.3, 0.1)         # mean and standard deviation of the start state
const REGULATOR_γ, REGULATOR_HORIZON = 0.9, 10

# Sample the next state; the reward -s² depends on the current state.
regulator_step(s, a, rng) = (s + a + REGULATOR_NOISE_SD * randn(rng), -s^2)

# Stochastic linear policy a ~ N(θ₁ s, θ₂²); |θ₂| + 1e-5 keeps the spread positive.
regulator_policy(θ, s, rng) = θ[1] * s + (abs(θ[2]) + 1e-5) * randn(rng)

# U(θ) without sampling: E[s²] follows (1 + θ₁)² E[s²] + θ₂² + 0.1² from step to step.
# ∇U(θ) by differentiating the same recursion (checked against Python in the parity test).
function regulator_exact_gradient(θ; d = REGULATOR_HORIZON)
    gain, sd = 1 + θ[1], abs(θ[2]) + 1e-5
    noise_var = sd^2 + REGULATOR_NOISE_SD^2
    ms, dms, grad = REGULATOR_START[1]^2 + REGULATOR_START[2]^2, zeros(2), zeros(2)
    for t in 1:d
        grad .-= REGULATOR_γ^(t - 1) .* dms
        dms = [2gain * ms + gain^2 * dms[1], gain^2 * dms[2] + 2sd * sign(θ[2])]
        ms = gain^2 * ms + noise_var
    end
    return grad
end

function regulator_exact_value(θ; d = REGULATOR_HORIZON)
    gain, noise_var = 1 + θ[1], (abs(θ[2]) + 1e-5)^2 + REGULATOR_NOISE_SD^2
    ms, u = REGULATOR_START[1]^2 + REGULATOR_START[2]^2, 0.0
    for t in 1:d
        u -= REGULATOR_γ^(t - 1) * ms
        ms = gain^2 * ms + noise_var
    end
    return u
end
