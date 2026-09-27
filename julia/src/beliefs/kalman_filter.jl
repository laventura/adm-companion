# Kalman filter (ADM §19.3) for a linear-Gaussian model:
#   T(s′ | s, a) = N(s′ | Ts s + Ta a, Σs)     O(o | s′) = N(o | Os s′, Σo)
# The belief is a Gaussian N(μb, Σb). Works with matrices, or with plain numbers in 1D.
struct LinearGaussianModel
    Ts; Ta; Σs; Os; Σo
end

# Predict: move the mean through the dynamics; the covariance grows by Σs.
function kalman_predict(μb, Σb, a, m::LinearGaussianModel)
    μp = m.Ts * μb + m.Ta * a
    Σp = m.Ts * Σb * m.Ts' + m.Σs
    return μp, Σp
end

# Correct: blend prediction and observation using the Kalman gain K.
function kalman_correct(μp, Σp, o, m::LinearGaussianModel)
    # A / B means A * inv(B), computed without forming the inverse.
    # (Os Σp Os' + Σo) is the covariance of the predicted observation.
    K = Σp * m.Os' / (m.Os * Σp * m.Os' + m.Σo)
    μb = μp + K * (o - m.Os * μp)     # move toward the observation by the innovation
    Σb = Σp - K * m.Os * Σp           # same as (I - K Os) Σp, but also works with plain numbers
    return μb, Σb, K
end

function kalman_update(μb, Σb, a, o, m::LinearGaussianModel)
    μp, Σp = kalman_predict(μb, Σb, a, m)
    μb′, Σb′, _ = kalman_correct(μp, Σp, o, m)
    return μb′, Σb′
end
