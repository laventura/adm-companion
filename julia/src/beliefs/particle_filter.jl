# Particle filter (ADM §19.6). The belief is a vector of sampled states.
#   sample_next(rng, s, a) draws s′ ~ T(· | s, a)
#   likelihood(o, a, s′)   returns O(o | a, s′)

function update_particles(particles, a, o, sample_next, likelihood, rng)
    moved = [sample_next(rng, s, a) for s in particles]    # predict
    w = [likelihood(o, a, s′) for s′ in moved]             # weight
    sum(w) > 0 || error("every particle has zero weight: the observation is impossible under this belief")
    keep = rand(rng, Categorical(w ./ sum(w)), length(moved))
    return moved[keep]                                     # resample
end
