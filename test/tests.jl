@testitem "compute_probs" begin
    using TrueAndErrorDependentModels
    using Test

    model = TEDM(; p = [0.1, 0.2, 0.3, 0.4], ϵ = [0.05, 0.10, 0.15, 0.20], p_rep = 0)
    probs = compute_probs(model)

    @test sum(model.p) ≈ 1
    @test sum(probs) ≈ 1

    true_probs = [
        0.086
        0.039
        0.037
        0.014
        0.039
        0.122
        0.014
        0.039
        0.037
        0.014
        0.187
        0.066
        0.014
        0.039
        0.066
        0.187
    ]
    @test probs ≈ true_probs atol = 0.002
end

@testitem "sum to 1" begin
    using Distributions
    using TrueAndErrorDependentModels
    using Test

    for _ ∈ 1:1000
        model = TEDM(;
            p = rand(Dirichlet([1, 1, 1, 1])),
            ϵ = rand(Uniform(0, 0.5), 4),
            p_rep = rand()
        )
        probs = compute_probs(model)

        @test sum(model.p) ≈ 1
        @test sum(probs) ≈ 1
    end
end

@testitem "critical test" begin
    using Distributions
    using TrueAndErrorDependentModels
    using Test

    for _ ∈ 1:1000
        model = TEDM(;
            p = [0,0,0,1],
            ϵ = rand(Uniform(0, 0.5), 4),
            p_rep = rand()
        )
        probs = compute_probs(model)

        @test probs[1] ≤ probs[16]
    end
end

@testitem "rand" begin
    using Random
    using TrueAndErrorDependentModels
    using Test

    Random.seed!(574)

    model = TEDM(; p = [0.1, 0.2, 0.3, 0.4], ϵ = [0.05, 0.10, 0.15, 0.20], p_rep = 0)
    n_trials = 100_000
    data = rand(model, n_trials)
    probs = data ./ n_trials

    @test sum(model.p) ≈ 1
    @test sum(probs) ≈ 1

    true_probs = [
        0.086
        0.039
        0.037
        0.014
        0.039
        0.122
        0.014
        0.039
        0.037
        0.014
        0.187
        0.066
        0.014
        0.039
        0.066
        0.187
    ]
    @test probs ≈ true_probs atol = 0.005
end

@testitem "constructors" begin
    using TrueAndErrorDependentModels
    using Test

    model1 = TEDM(; p = [0.1, 0.2, 0.3, 0.4], ϵ = [0.05, 0.10, 0.15, 0.20], p_rep = 0)
    model2 = TEDM([0.1, 0.2, 0.3, 0.4], [0.05, 0.10, 0.15, 0.20], 0)
    @test model1 == model2
end
