############################################################################################
# Differentiation - Benchmark Model
modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
objectiveExample = Objective(modelExample, (data1, data2, data3, _idx))

@testset "AutoDiffContainer - Log Objective Results" begin
    tune_fwd = AutomaticDiffTune(:ForwardDiff, objectiveExample)
    fwd = DiffObjective(objectiveExample, tune_fwd)
    theta_unconstrained = randn(_RNG, length(modelExample))
    ## Compute logdensity
    log_density(objectiveExample)
    theta_unconstrained2 = deepcopy(theta_unconstrained)
    #!NOTE: 10th parameter in likelihood for example, so is not compiled away in Reverse Tape
    theta_unconstrained2[10] = Inf
    ## Check if result is finite
    _ld_fin = log_density(objectiveExample, copy(theta_unconstrained))
    _ld_inf = log_density(objectiveExample, copy(theta_unconstrained2))
    _ld_inf_fault = log_density(objectiveExample, copy(theta_unconstrained))
    _ld_inf_fault.θᵤ[10] = Inf

    @test ModelWrappers.checkfinite(theta_unconstrained)
    @test !ModelWrappers.checkfinite(theta_unconstrained2)

    @test ModelWrappers.checkfinite(_ld_fin)
    @test !ModelWrappers.checkfinite(_ld_inf)
    @test !ModelWrappers.checkfinite(_ld_inf_fault)

    @test ModelWrappers.checkfinite(_ld_inf, _ld_fin) #Infinite to Finite
    @test !ModelWrappers.checkfinite(_ld_fin, _ld_inf) #Finite to Infinite

    @test ModelWrappers.checkfinite(-Inf, 10.0, _ld_fin) #Infinite to Finite
    @test !ModelWrappers.checkfinite(10.0, -Inf, _ld_fin) #Finite to Infinite
    @test !ModelWrappers.checkfinite(-Inf, 10.0, _ld_inf) #Finite to Infinite


    #!ToDo: Check for correct error message in case logdensity cannot be evaluated
    ModelWrappers.checkfinite(objectiveExample, theta_unconstrained)
    ModelWrappers.checkfinite(objectiveExample, _ld_fin)
    ModelWrappers.checkfinite(objectiveExample, _ld_inf, _ld_fin)
    ModelWrappers.checkfinite(objectiveExample, -Inf, 10.0, _ld_fin)

    err = ObjectiveError(objectiveExample, -Inf, theta_unconstrained)
    @test isa(err, ArgCheck.Exception)

end


@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Vectorized Model" begin
    ## Assign DiffTune
    tune_fwd = AutomaticDiffTune(:ForwardDiff, objectiveExample)
    tune_rd = AutomaticDiffTune(:ReverseDiff, objectiveExample)
    tune_zyg = AutomaticDiffTune(:Zygote, objectiveExample)
    fwd = DiffObjective(objectiveExample, tune_fwd)
    rd = DiffObjective(objectiveExample, tune_rd)
    zyg = DiffObjective(objectiveExample, tune_zyg)
    theta_unconstrained = randn(_RNG, length(modelExample))
    ## Compute logdensity
    log_density(objectiveExample)
    theta_unconstrained2 = deepcopy(theta_unconstrained)
    #!NOTE: 10th parameter in likelihood for example, so is not compiled away in Reverse Tape
    theta_unconstrained2[10] = Inf
    _ld = log_density(objectiveExample, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(fwd, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(rd, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)
    _ld = log_density(zyg, theta_unconstrained2)
    @test isinf(_ld.ℓθᵤ)

    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld3 = log_density(zyg, theta_unconstrained)
    _ld1 = _log_density(objectiveExample, tune_fwd, theta_unconstrained)
    _ld2 = _log_density(objectiveExample, tune_rd, theta_unconstrained)
    _ld3 = _log_density(objectiveExample, tune_zyg, theta_unconstrained)
    ## Compute Diffresult
    log_density_and_gradient(objectiveExample, tune_fwd, theta_unconstrained2)
    _grad = log_density_and_gradient(fwd, theta_unconstrained2)
    @test isinf(_grad.ℓθᵤ)
    log_density_and_gradient(objectiveExample, tune_rd, theta_unconstrained2)
    _grad = log_density_and_gradient(rd, theta_unconstrained2)
    #!TODO: Need an example where tape does not compile parameter in infi
    @test isinf(_grad.ℓθᵤ)
    log_density_and_gradient(objectiveExample, tune_zyg, theta_unconstrained2)
    _grad = log_density_and_gradient(zyg, theta_unconstrained2)
    @test isinf(_grad.ℓθᵤ)
    _grad1 = _log_density_and_gradient(objectiveExample, tune_fwd, theta_unconstrained)
    _grad2 = _log_density_and_gradient(objectiveExample, tune_rd, theta_unconstrained)
    _grad3 = _log_density_and_gradient(objectiveExample, tune_zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compute manual call ~ Already checked for equality
    ld = objectiveExample(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(objectiveExample, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(_grad1[2] - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(_grad2[2] - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(_grad3[2] - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    ## Checks
    _output = check_gradients(_RNG, objectiveExample, [:ForwardDiff, :ReverseDiff, :Zygote]; printoutput = false)
    @test sum(abs.(_output.ℓobjective_gradient_diff)) ≈ 0 atol = _TOL
    ## Update DiffTune
    ModelWrappers.update(tune_fwd, objectiveExample)
    ModelWrappers.update(tune_rd, objectiveExample)
    ModelWrappers.update(tune_zyg, objectiveExample)
    ## Config DiffTune
    theta_unconstrained2 = randn(_RNG, length(objectiveExample))
    ModelWrappers._config(ModelWrappers.ADForward(), objectiveExample, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADReverse(), objectiveExample, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADReverseUntaped(), objectiveExample, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADZygote(), objectiveExample, theta_unconstrained2)
end

############################################################################################
# Differentiation - Lower dimensions
modelLowerDim = ModelWrapper(LowerDims(), _val_lowerdims)
objectiveLowerDim = Objective(modelLowerDim, nothing)

@testset "AutoDiffContainer - Log Objective AutoDiff compatibility - Lower dimensions" begin
    ## Assign DiffTune
    autodiff_fd = AutomaticDiffTune(:ForwardDiff, objectiveLowerDim)
    autodiff_rd = AutomaticDiffTune(:ReverseDiff, objectiveLowerDim)
    autodiff_zyg = AutomaticDiffTune(:Zygote, objectiveLowerDim)
    fwd = DiffObjective(objectiveLowerDim, autodiff_fd)
    rd = DiffObjective(objectiveLowerDim, autodiff_rd)
    zyg = DiffObjective(objectiveLowerDim, autodiff_zyg)
    theta_unconstrained = randn(_RNG, length(objectiveLowerDim))
    ## Compute Diffresult
    ld1 = log_density(fwd, theta_unconstrained)
    ld2 = log_density(rd, theta_unconstrained)
    ld3 = log_density(zyg, theta_unconstrained)
    _ld1 = _log_density(objectiveLowerDim, autodiff_fd, theta_unconstrained)
    _ld2 = _log_density(objectiveLowerDim, autodiff_rd, theta_unconstrained)
    _ld3 = _log_density(objectiveLowerDim, autodiff_zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compute manual call ~ Already checked for equality
    ld = objectiveLowerDim(theta_unconstrained)
    grad_mod_fd = ForwardDiff.gradient(objectiveLowerDim, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveLowerDim, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(objectiveLowerDim, theta_unconstrained)[1]
    ## Compare results
    @test ld - ld1.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld2.ℓθᵤ ≈ 0 atol = _TOL
    @test ld - ld3.ℓθᵤ ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad1.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_rd - grad2.∇ℓθᵤ)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_zy - grad3.∇ℓθᵤ)) ≈ 0 atol = _TOL
    ## Checks
    _output = check_gradients(_RNG, objectiveLowerDim, [:ForwardDiff, :ReverseDiff, :Zygote]; printoutput = false)
    @test sum(abs.(_output.ℓobjective_gradient_diff)) ≈ 0 atol = _TOL
    ## Results
    ℓDensityResult(objectiveLowerDim, theta_unconstrained)
    ℓDensityResult(objectiveLowerDim)
    ℓDensityResult(fwd)
    ℓDensityResult(rd)
    ℓDensityResult(zyg)
    ℓGradientResult(grad1.θᵤ , grad1.ℓθᵤ , grad1.∇ℓθᵤ)
    ## Update DiffTune
    ModelWrappers.update(autodiff_fd, objectiveLowerDim)
    ModelWrappers.update(autodiff_rd, objectiveLowerDim)
    ModelWrappers.update(autodiff_zyg, objectiveLowerDim)
    ## Config DiffTune
    theta_unconstrained2 = randn(_RNG, length(objectiveLowerDim))
    ModelWrappers._config(ModelWrappers.ADForward(), objectiveLowerDim, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADReverse(), objectiveLowerDim, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADReverseUntaped(), objectiveLowerDim, theta_unconstrained2)
    ModelWrappers._config(ModelWrappers.ADZygote(), objectiveLowerDim, theta_unconstrained2)
end

############################################################################################
# Differentiation - Float32
modelExample2 = ModelWrapper(ExampleModel(), _val_examplemodel, (;), FlattenDefault(; output = Float32))
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
    ld3 = log_density(zyg, theta_unconstrained)
    grad1 = log_density_and_gradient(fwd, theta_unconstrained)
    grad2 = log_density_and_gradient(rd, theta_unconstrained)
    grad22 = log_density_and_gradient(rd2, theta_unconstrained)
    grad3 = log_density_and_gradient(zyg, theta_unconstrained)
    ## Compare types
    @test ld1.ℓθᵤ isa T && eltype(ld1.θᵤ) == T
    @test ld2.ℓθᵤ isa T && eltype(ld2.θᵤ) == T
    @test ld22.ℓθᵤ isa T && eltype(ld22.θᵤ) == T
    @test ld3.ℓθᵤ isa T && eltype(ld3.θᵤ) == T

    @test grad1.ℓθᵤ isa T && eltype(grad1.θᵤ) == eltype(grad1.∇ℓθᵤ) == T
    @test grad2.ℓθᵤ isa T && eltype(grad2.θᵤ) == eltype(grad2.∇ℓθᵤ) == T
    @test grad22.ℓθᵤ isa T && eltype(grad22.θᵤ) == eltype(grad22.∇ℓθᵤ) == T
    @test grad3.ℓθᵤ isa T && eltype(grad3.θᵤ) == eltype(grad3.∇ℓθᵤ) == T
end

############################################################################################
#Tune Analytic
function fun1(objective::Objective{<:ModelWrapper{M}}, θᵤ::AbstractVector{T}) where {M<:ExampleModel, T<:Real}
    return zeros(size(θᵤ))
end
θᵤ = randn(_RNG, length(objectiveExample))
fun1(objectiveExample, θᵤ)
@testset "AnalyticDiffTune - " begin
    tune_analytic = AnalyticalDiffTune(fun1)
    ModelWrappers.update(tune_analytic, objectiveExample)
    _ld = _log_density(objectiveExample, tune_analytic, θᵤ)
    _ldg =_log_density_and_gradient(objectiveExample, tune_analytic, θᵤ)
    @test _ld == _ldg[1]
    _ldgresult = log_density_and_gradient(objectiveExample, tune_analytic, θᵤ)
    @test _ld == _ldgresult.ℓθᵤ
    @test all(_ldgresult.θᵤ .== θᵤ)
end
