############################################################################################
# Differentiation - Benchmark Model
modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
objectiveExample = Objective(modelExample, (data1, data2, data3, _idx))

θᵤ = randn(length(objectiveExample))
tune_fwd = AutomaticDiffTune(:ForwardDiff, objectiveExample)
tune_rd = AutomaticDiffTune(:ReverseDiff, objectiveExample)
tune_zyg = AutomaticDiffTune(:Zygote, objectiveExample)
fwd = DiffObjective(objectiveExample, tune_fwd)
rd = DiffObjective(objectiveExample, tune_rd)
zyg = DiffObjective(objectiveExample, tune_zyg)

@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Vectorized Model" begin
    theta_unconstrained = randn(length(modelExample))
    ## Compute Diffresult
    _grad1 = _log_density_and_gradient(objectiveExample, tune_fwd, theta_unconstrained)
    _grad2 = _log_density_and_gradient(objectiveExample, tune_rd, theta_unconstrained)
#    _grad3 = _log_density_and_gradient(objectiveExample, tune_zyg, theta_unconstrained)
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
#    ld3 = log_density(zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
#    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compute manual call ~ Already checked for equality
    ld = objectiveExample(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveExample, theta_unconstrained)
#    grad_mod_zy = Zygote.gradient(objectiveExample, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
#    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(_grad1[2] - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(_grad2[2] - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(_grad3[2] - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
end

############################################################################################
# Differentiation - Lower dimensions
modelLowerDim = ModelWrapper(LowerDims(), _val_lowerdims)
objectiveLowerDim = Objective(modelLowerDim, nothing)

fwd = DiffObjective(objectiveLowerDim, AutomaticDiffTune(:ForwardDiff, objectiveLowerDim))
rd = DiffObjective(objectiveLowerDim, AutomaticDiffTune(:ReverseDiff, objectiveLowerDim))
zyg = DiffObjective(objectiveLowerDim, AutomaticDiffTune(:Zygote, objectiveLowerDim))

@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Lower dimensions" begin
    theta_unconstrained = randn(length(objectiveLowerDim))
    ## Compute Diffresult
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
#    ld3 = log_density(zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
#    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compute manual call ~ Already checked for equality
    ld = objectiveLowerDim(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveLowerDim, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveLowerDim, theta_unconstrained)
#    grad_mod_zy = Zygote.gradient(objectiveLowerDim, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
#    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
end

############################################################################################
# Differentiation - Float32
modelExample2 = ModelWrapper(ExampleModel(), _val_examplemodel, FlattenDefault(; output = Float32))
objectiveExample2 = Objective(modelExample2, (data1, data2, data3, _idx))
objectiveExample2(objectiveExample2.model.val)

fwd = DiffObjective(objectiveExample2, AutomaticDiffTune(:ForwardDiff, objectiveExample2))
rd = DiffObjective(objectiveExample2, AutomaticDiffTune(:ReverseDiff, objectiveExample2))
rd2 = DiffObjective(objectiveExample2, AutomaticDiffTune(:ReverseDiffUntaped, objectiveExample2))
zyg = DiffObjective(objectiveExample2, AutomaticDiffTune(:Zygote, objectiveExample2))

@testset "AutoDiffContainer - Float32 compatibility" begin
    T = Float32
    theta_unconstrained = randn(T, length(objectiveExample2))
    ## Compute Diffresult
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld22 = log_density(rd2, theta_unconstrained)
#    ld3 = log_density(zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad22 = log_density_and_gradient(rd2, theta_unconstrained)
#    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compare types
    @test ld1.ℓθᵤ isa T && eltype(ld1.θᵤ) == T
    @test ld2.ℓθᵤ isa T && eltype(ld2.θᵤ) == T
    @test ld22.ℓθᵤ isa T && eltype(ld22.θᵤ) == T
#    @test ld3.ℓθᵤ isa T && eltype(ld3.θᵤ) == T

    @test grad1.ℓθᵤ isa T && eltype(grad1.θᵤ) == eltype(grad1.∇ℓθᵤ) == T
    @test grad2.ℓθᵤ isa T && eltype(grad2.θᵤ) == eltype(grad2.∇ℓθᵤ) == T
    @test grad22.ℓθᵤ isa T && eltype(grad22.θᵤ) == eltype(grad22.∇ℓθᵤ) == T
#    @test grad3.ℓθᵤ isa T && eltype(grad3.θᵤ) == eltype(grad3.∇ℓθᵤ) == T
end
