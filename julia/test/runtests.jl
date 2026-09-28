# Run with: julia --project=julia -e 'using Pkg; Pkg.test()'
# Also pins the numbers quoted in the Ch 19 page.
using ADM, LinearAlgebra, Random, Test

@testset "discrete filter: tram, first observation" begin
    n, tunnels = 12, [3, 4, 5, 9, 11]
    T = zeros(n, 1, n)
    for s in 1:n, (step, p) in ((0, 0.1), (1, 0.8), (2, 0.1))
        T[s, 1, mod1(s + step, n)] += p
    end
    O = zeros(1, n, 2)                       # o = 1 light, o = 2 dark
    for s in 1:n
        p_dark = s in tunnels ? 0.9 : 0.2
        O[1, s, :] = [1 - p_dark, p_dark]
    end
    b = fill(1 / n, n)
    @test predict_discrete(b, T, 1) ≈ b      # a uniform belief stays uniform on a ring
    b′ = update_discrete(b, T, O, 1, 2)
    @test sum(b′) ≈ 1
    @test b′[3] ≈ 0.9 / 5.9 && b′[1] ≈ 0.2 / 5.9
    @test_throws ArgumentError correct_discrete([1.0, 0.0], zeros(1, 2, 1), 1, 1)
end

@testset "Kalman filter" begin
    kayak = LinearGaussianModel(1.0, 1.0, 1.0, 1.0, 4.0)
    μp, Σp = kalman_predict(0.0, 9.0, 2.0, kayak)
    @test (μp, Σp) == (2.0, 10.0)
    μb, Σb, K = kalman_correct(μp, Σp, 3.5, kayak)
    @test K ≈ 10 / 14 && μb ≈ 2 + 15 / 14 && Σb ≈ 20 / 7

    Σ = 9.0                                   # steady state: Σ² + Σ - 4 = 0
    for _ in 1:50
        _, Σ = kalman_update(0.0, Σ, 0.0, 0.0, kayak)
    end
    @test Σ ≈ (sqrt(17) - 1) / 2

    buoy = LinearGaussianModel([1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1], zeros(4, 2),
                               Diagonal([0.01, 0.01, 0.05, 0.05]), [1 0 0 0; 0 1 0 0], 4.0I(2))
    μ, Σ2 = kalman_update(zeros(4), 10.0I(4), zeros(2), [1.0, -2.0], buoy)
    @test Σ2 ≈ Σ2' && all(eigvals(Symmetric(Σ2)) .> 0)
end

@testset "particle filter matches Kalman on a linear-Gaussian problem" begin
    rng = MersenneTwister(19)
    particles = 3 .* randn(rng, 20_000)       # prior N(0, 9)
    sample_next(rng, s, a) = s + a + randn(rng)
    likelihood(o, a, s′) = exp(-(o - s′)^2 / (2 * 4))
    p = update_particles(particles, 2.0, 3.5, sample_next, likelihood, rng)
    @test isapprox(sum(p) / length(p), 2 + 15 / 14; atol=0.06)
    @test isapprox(sum(abs2, p .- sum(p) / length(p)) / length(p), 20 / 7; atol=0.15)
    @test_throws ErrorException update_particles([0.0], 0.0, 0.0, (r, s, a) -> s, (o, a, s) -> 0.0, rng)
end

@testset "Hex World" begin
    m = hex_world()
    @test size(m.T) == (25, 6, 25) && m.γ == 0.9 && m.states[end] == "terminal"
    @test all(sum(m.T, dims = 3) .≈ 1)
    @test ADM.hex_neighbor((1, 8), "SE") == (2, 9)
    i, east = findfirst(==((0, 0)), m.states), findfirst(==("E"), m.actions)
    @test m.R[i, east] ≈ -0.15
    @test size(straight_line_hex_world().T) == (8, 6, 8)
end

@testset "Bayes net chain rule" begin
    c = commute_network()
    @test probability(c, Dict(:S => 2, :R => 1, :J => 2, :L => 2, :A => 2)) ≈ 0.1 * 0.8 * 0.7 * 0.5 * 0.95
end

@testset "Mountain car" begin
    s = mountain_car_step([-0.5, 0.0], 1)
    @test s ≈ [-0.5 + (0.001 - 0.0025cos(-1.5)), 0.001 - 0.0025cos(-1.5)]
    @test mountain_car_step([0.59, 0.07], 1)[1] == 0.6 && mountain_car_goal([0.6, 0.0])
    @test mountain_car_step([-1.2, -0.07], -1)[1] == -1.2   # position clamped at the left wall
    @test mountain_car_step([-1.0, 0.07], 1)[2] == 0.07     # speed clamped at its maximum
end

@testset "show_source" begin
    out = sprint(io -> show_source("beliefs/discrete_filter.jl", io))
    @test startswith(out, "```julia\n") && endswith(rstrip(out), "```")
end

@testset "crying baby and exact POMDP planning (Ch 20)" begin
    p = crying_baby()
    @test all(sum(p.T; dims = 3) .≈ 1) && all(sum(p.O; dims = 3) .≈ 1)
    leaf(a) = ConditionalPlan(a)
    # ignore; then feed if crying, else ignore; then feed if crying, else ignore.
    π = ConditionalPlan(3, [ConditionalPlan(1, [leaf(1), leaf(3)]), ConditionalPlan(3, [leaf(1), leaf(3)])])
    α = alphavector(p, π)
    @test combine_alpha(p, 3, [alphavector(p, s) for s in π.subplans]) ≈ α
    @test evaluate_plan(p, leaf(1), 2) == -15.0
    # The simplex LP agrees with a brute-force search over beliefs.
    for (αt, Γt) in (([1.0, 0.0], [[0.0, 1.0], [0.6, 0.6]]), ([0.5, 0.5], [[0.0, 1.0], [0.6, 0.6]]),
                     ([2.0, 1.0, 0.0], [[0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]))
        b = find_maximal_belief(αt, Γt)
        grid = [[x, y, 1 - x - y][1:length(αt)] for x in 0:0.01:1 for y in (length(αt) == 3 ? (0:0.01:1-x) : (1 - x,))]
        gap = maximum(minimum((αt .- g) ⋅ bb for g in Γt) for bb in grid)
        @test (b === nothing) == (gap <= 1e-9)
        b === nothing || @test minimum((αt .- g) ⋅ b for g in Γt) ≈ gap atol = 0.02
    end
    # Value iteration's pruned envelope equals the envelope of every 3-step plan.
    hist, counts = value_iteration_pomdp(p, 30)
    all1 = [ConditionalPlan(a) for a in 1:3], [p.R[:, a] for a in 1:3]
    all2 = expand(all1..., p)
    _, Γ3 = expand(all2..., p)
    @test length(Γ3) == 3^7
    grid = [[1 - x, x] for x in 0:0.001:1]
    @test maximum(abs(utility(Γ3, b) - utility(hist[3][2], b)) for b in grid) < 1e-9
    # Long-horizon policy: ignore below about P(hungry) = 0.28, feed above; never sing.
    pl, Γ = hist[30]
    root = [pl[argmax([α ⋅ b for α in Γ])].a for b in grid]
    @test Set(root) == Set([1, 3])
    @test 0.27 < grid[findfirst(==(1), root)][2] < 0.29
end
