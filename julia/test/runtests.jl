# Run with: julia --project=julia -e 'using Pkg; Pkg.test()'
# Also pins the numbers quoted in the Ch 19 page.
using ADM, LinearAlgebra, Random, Statistics, Test

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

@testset "offline belief state planning (Ch 21)" begin
    p = crying_baby()
    Γ_opt = value_iteration_pomdp(p, 200)[1][end][2]      # 200 steps: within 1e-7 of infinite
    grid = [[1 - x, x] for x in 0:0.01:1]
    opt(b) = utility(Γ_opt, b)
    q, f = qmdp(p; k_max = 300), fast_informed_bound(p; k_max = 300)
    @test all(utility(q, b) >= utility(f, b) - 1e-9 >= opt(b) - 2e-9 for b in grid)   # QMDP ≥ FIB ≥ U*
    @test all(utility(blind_lowerbound(p; k_max = 300), b) <= opt(b) + 1e-9 for b in grid)
    @test all(baws_lowerbound(p) ⋅ b <= opt(b) for b in grid)
    B = [[1 - x, x] for x in 0:0.25:1]
    Γp = pbvi(p, B; k_max = 100)
    @test all(-0.01 < utility(Γp, b) - opt(b) <= 1e-9 for b in grid)                 # a tight lower bound
    Γr = perseus(p, B, MersenneTwister(1); k_max = 100)
    @test all(utility(Γr, b) <= opt(b) + 1e-9 for b in grid)
    V = sawtooth_iteration(p, Dict(e => utility(f, e) for e in ([1.0, 0.0], [0.0, 1.0])),
                           [[1 - x, x] for x in 0:0.2:1]; k_max = 10)
    @test all(sawtooth(V, b) >= opt(b) - 1e-9 for b in grid)
    _, U = triangulated_iteration(p, 10; k_max = 150)
    @test all(0 <= U(b) - opt(b) < 0.2 for b in grid)
    _, _, gaps = sawtooth_heuristic_search(p, [1.0, 0.0])
    @test gaps[end] < 0.01 && gaps[end] < gaps[1] / 1000
    # Freudenthal interpolation reconstructs the belief from grid vertices.
    for b in ([0.2, 0.3, 0.5], [0.05, 0.9, 0.05], [1 / 3, 1 / 3, 1 / 3])
        verts, λ = freudenthal(b, 4)
        @test sum(λ) ≈ 1 && all(λ .>= -1e-12) && sum(l .* v for (v, l) in zip(verts, λ)) ≈ b
        @test all(all(isinteger(round(4x; digits = 9)) for x in v) for (v, l) in zip(verts, λ) if l > 1e-12)
    end
    # The fork: upper-bound policies wait forever; point-based planning asks.
    fk = fork()
    b0 = [0.5, 0.5]
    for Γ in (qmdp(fk; k_max = 300), fast_informed_bound(fk; k_max = 300))
        @test greedy_belief(fk, b -> utility(Γ, b), b0).a == 1          # wait, and b0 never changes
    end
    Γf = pbvi(fk, [[1 - x, x] for x in range(0, 1, length = 21)]; k_max = 300)
    @test greedy_belief(fk, b -> utility(Γf, b), b0).a == 2              # ask
    _, Uf = triangulated_iteration(fk, 20; k_max = 400)
    @test 0 <= Uf(b0) - utility(Γf, b0) < 1e-4                          # the bounds pinch
end

@testset "online belief state planning (Ch 22)" begin
    p = crying_baby()
    b0 = [0.5, 0.5]
    grid = [[1 - x, x] for x in 0:0.05:1]
    Γ_opt = value_iteration_pomdp(p, 200)[1][end][2]
    opt(b) = utility(Γ_opt, b)
    f = fast_informed_bound(p; k_max = 300)
    Uf(b) = utility(f, b)
    # Forward search: the error shrinks at least as fast as γ^d times the leaf's error.
    leaf_err = maximum(abs(Uf(b) - opt(b)) for b in grid)
    for d in 0:3
        c = Ref(0)
        @test all(abs(forward_search_belief(p, b, d, Uf; count = c).u - opt(b)) <= p.γ^d * leaf_err + 1e-9
                  for b in grid)
        @test c[] == length(grid) * (6^(d + 1) - 1) ÷ 5        # 3 actions × 2 observations per level
    end
    # Branch and bound returns forward search's answer; tight bounds prune.
    lo = pbvi(p, [[1 - x, x] for x in 0:0.25:1]; k_max = 100)
    Ulo(b) = utility(lo, b)
    Qhi(b, a) = lookahead_belief(p, Uf, b, a)
    c1, c2 = Ref(0), Ref(0)
    for b in grid
        r1, r2 = forward_search_belief(p, b, 3, Ulo; count = c1), branch_and_bound_belief(p, b, 3, Ulo, Qhi; count = c2)
        @test r1.a == r2.a && r1.u ≈ r2.u
    end
    @test c2[] < c1[] / 5
    # Sparse sampling averages toward forward search, from above (max of noisy means).
    fs = forward_search_belief(p, b0, 2, Uf).u
    us = [sparse_sampling_belief(p, b0, 2, 16, Uf, MersenneTwister(i)).u for i in 1:200]
    @test 0 < mean(us) - fs < 1.5
    # Determinized particles: inverse-CDF on P(s′, o | sated, ignore) = (0.09, 0.81, 0.08, 0.02).
    Φ = [0.05 0.62 0.95 0.999]
    @test [successor(p, Φ, DeterminizedParticle(1, 1, j), 3) for j in 1:4] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    ds = [determinized_search(p, rand(MersenneTwister(i), 256, 4), determinized_belief(b0, 256, MersenneTwister(i)), 4,
                               ps -> Uf(particle_belief(p, ps))).u for i in 1:20]
    @test abs(mean(ds) - forward_search_belief(p, b0, 4, Uf).u) < 0.5
    # Gap heuristic search with true bounds keeps them and closes the gap at b0.
    g = gap_heuristic_search(p, b0, b -> utility(blind_lowerbound(p; k_max = 300), b), Uf; k_max = 300, d_max = 20)
    @test g.a == 1 && g.history[end][2] - g.history[end][1] < 0.1
    @test all(g.lo[b] <= opt(b) + 1e-9 && opt(b) <= g.hi[b] + 1e-9 for b in keys(g.lo))
    # The fork: two steps of search on QMDP fix its waiting; MCTS asks.
    fk = fork()
    uq(b) = utility(qmdp(fk; k_max = 300), b)
    @test forward_search_belief(fk, b0, 1, uq).a == 1 && forward_search_belief(fk, b0, 2, uq).a == 2
    @test forward_search_belief(fk, b0, 2, b -> 0.0).a == 1 && forward_search_belief(fk, b0, 3, b -> 0.0).a == 2
    @test count(i -> HistoryMCTS(fk; d = 20, c = 50.0)(b0, 1000, MersenneTwister(i)) == 2, 1:20) >= 18
end

@testset "controllers (Ch 23)" begin
    p, fk, b0 = crying_baby(), fork(), [0.5, 0.5]
    opt = utility(value_iteration_pomdp(p, 200)[1][end][2], b0)
    # Two 2-node controllers: feed on crying; the wiser one never feeds twice in a row.
    react = deterministic_controller(p, [3, 1], [[2, 1], [2, 1]])
    wise = deterministic_controller(p, [3, 1], [[2, 1], [1, 1]])
    @test controller_value(p, wise, b0).u ≈ opt atol = 1e-6
    @test controller_value(p, react, b0).u < opt - 0.5
    # Evaluation agrees with simulation (5,000 runs from the feed node).
    rets = [run_controller(p, wise, 2, rand(MersenneTwister(i), 1:2), 100, MersenneTwister(i))[1] for i in 1:5000]
    @test abs(mean(rets) - controller_value(p, wise, b0).u) < 3 * std(rets) / sqrt(5000)
    # The fork: counting answers (5 nodes) is optimal; 4-node brute force gets 15.33, 3 nodes only wait.
    counter = deterministic_controller(fk, [2, 2, 2, 4, 3], [[3, 2], [1, 4], [5, 1], [1, 1], [1, 1]])
    @test controller_value(fk, counter, b0).u ≈ 27.1149 atol = 1e-3
    @test best_deterministic_controller(fk, 3, b0).u ≈ 0 atol = 1e-9
    @test best_deterministic_controller(p, 2, b0).u ≈ opt atol = 1e-6
    # Policy iteration reaches the optimum on both problems.
    _, h = controller_policy_iteration(p, b0; k_max = 4)
    @test h[end][2] ≈ opt atol = 1e-6
    _, hf = controller_policy_iteration(fk, b0; k_max = 4)
    @test hf[end][2] ≈ controller_value(fk, counter, b0).u atol = 1e-6
    # The gradient matches finite differences; projection lands on the simplex.
    c = random_controller(p, 2, MersenneTwister(3))
    g = controller_gradient(p, c, b0)
    ψ2 = copy(c.ψ); ψ2[2, 1] += 1e-6
    @test (controller_gradient(p, Controller(ψ2, c.η), b0).J - g.J) / 1e-6 ≈ g.ψ[2, 1] rtol = 1e-4
    η2 = copy(c.η); η2[1, 3, 1, 2] += 1e-6
    @test (controller_gradient(p, Controller(c.ψ, η2), b0).J - g.J) / 1e-6 ≈ g.η[1, 3, 1, 2] rtol = 1e-4
    @test project_simplex([0.7, 0.5, -0.1]) ≈ [0.6, 0.4, 0.0]
    # Gradient ascent: optimal for the crying baby from most starts; stuck waiting at the fork.
    @test count(i -> controller_gradient_ascent(p, b0, 2, MersenneTwister(i); α = 0.003, k_max = 3000)[2][end] > opt - 0.01, 1:10) >= 6
    @test abs(controller_gradient_ascent(fk, b0, 5, MersenneTwister(1); α = 0.003, k_max = 3000)[2][end]) < 1e-3
end
