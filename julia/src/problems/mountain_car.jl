# Mountain car (ADM Appendix F.4). Mirrors adm/problems/mountain_car.py;
# tests/test_problems_parity.py checks that both languages move the car identically.
# State s = [x, v]: position in [-1.2, 0.6], speed in [-0.07, 0.07].
# Actions: -1 accelerate left, 0 coast, +1 accelerate right. Reward -1 per step until x = 0.6.
const MOUNTAIN_CAR_X = (-1.2, 0.6)
const MOUNTAIN_CAR_V = (-0.07, 0.07)
const MOUNTAIN_CAR_ACTIONS = (-1, 0, 1)

# Speed changes first (engine plus gravity, which pulls along the slope cos(3x)),
# then position moves by the new speed; both are clamped to the bounds.
function mountain_car_step(s, a)
    x, v = s
    v′ = clamp(v + 0.001a - 0.0025cos(3x), MOUNTAIN_CAR_V...)
    x′ = clamp(x + v′, MOUNTAIN_CAR_X...)
    return [x′, v′]
end

mountain_car_goal(s) = s[1] >= MOUNTAIN_CAR_X[2]
