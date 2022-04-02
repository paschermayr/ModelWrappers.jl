################################################################################
dat = randn(100)
μ₀ = 1.0
σ₀ = 2.0
################################################################################
# Create custom Model ~ Name it to avoid name collision
_param = (μ=Param(μ₀, Distributions.Normal()), σ=Param(σ₀, Distributions.Exponential()))
struct SossBenchmark <: ModelName end
modelSossBM = ModelWrapper(SossBenchmark(), _param)
obectiveSossBM = Objective(modelSossBM, dat)

function (objective::Objective{<:ModelWrapper{SossBenchmark}})(θ::NamedTuple)
    lp =
        Distributions.logpdf(Distributions.Normal(), θ.μ) +
        Distributions.logpdf(Distributions.Exponential(), θ.σ)
    ll = sum(
        Distributions.logpdf(Distributions.Normal(θ.μ, θ.σ), objective.data[iter]) for
        iter in eachindex(objective.data)
    )
    return lp + ll
end


################################################################################
#!NOTE: Remove Soss dependency from ModelWrappers and make separete BaytesSoss, so heavy deps. removed
# Create Soss Model
#=
m = @model n begin
    μ ~ Distributions.Normal()
    σ ~ Distributions.Exponential()
    data ~ iid(n)(Distributions.Normal(μ, σ))
    return (; data)
end
post = m((μ=μ₀, σ=σ₀, n=length(dat))) | (data=dat,)
model_soss = ModelWrapper(post)
objective_soss = Objective(model_soss)

################################################################################
#=
Goal for Soss Model:
        -> Constraints/Transforms/logdensities need to be the same for manual implementation and Soss model.
=#
@testset "Objective - Model compatibility" begin
    # Basic Functionality
    @test length(modelSossBM) == length(model_soss)
    @test length(obectiveSossBM) == length(objective_soss)
    @test _checkkeys(modelSossBM.val, model_soss.val)

    theta_unconstrained = randn(length(modelSossBM))
    _θ1, _ = flatten(
        unflatten_constrain(modelSossBM, theta_unconstrained), modelSossBM.info.constraint
    )
    _θ2, _ = flatten(
        unflatten_constrain(model_soss, theta_unconstrained), model_soss.info.constraint
    )
    @test sum(abs.(_θ1 - _θ2)) ≈ 0 atol = _TOL
    # Soss Posterior == Log likelihood + Prior
    lpost = obectiveSossBM(theta_unconstrained)
    lpost_soss = objective_soss(theta_unconstrained)
    @test abs(lpost - lpost_soss) ≈ 0 atol = _TOL
end

################################################################################
#=
Goal for lobjective:
        -> Should work with common AutoDiff packages
=#
@testset "Objective - Log Objective AutoDiff compatibility" begin
    #!NOTE - Zygote will be added back once Package is a bit more mature, as it adds a lot of time to tests
    theta_unconstrained = randn(length(modelSossBM))

    grad_mod_fd = ForwardDiff.gradient(obectiveSossBM, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(obectiveSossBM, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(obectiveSossBM, theta_unconstrained)[1]

    grad_modsoss_fd = ForwardDiff.gradient(objective_soss, theta_unconstrained)
    grad_modsoss_rd = ReverseDiff.gradient(objective_soss, theta_unconstrained)
    grad_modsoss_zy = Zygote.gradient(objective_soss, theta_unconstrained)[1]

    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_modsoss_fd - grad_modsoss_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_modsoss_fd - grad_modsoss_zy)) ≈ 0 atol = _TOL
end

_tagged = Tagged(modelSossBM, :σ)
obectiveSossBM2 = Objective(modelSossBM, dat, _tagged)
objective_soss2 = Objective(model_soss, _tagged)
@testset "Objective - Log Objective AutoDiff compatibility with subset of Model parameter" begin
    #!NOTE - Zygote will be added back once Package is a bit more mature, as it adds a lot of time to tests
    theta_unconstrained = randn(length(obectiveSossBM2.tagged))

    grad_mod_fd = ForwardDiff.gradient(obectiveSossBM2, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(obectiveSossBM2, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(obectiveSossBM2, theta_unconstrained)[1]

    grad_modsoss_fd = ForwardDiff.gradient(objective_soss2, theta_unconstrained)
    grad_modsoss_rd = ReverseDiff.gradient(objective_soss2, theta_unconstrained)
    grad_modsoss_zy = Zygote.gradient(objective_soss2, theta_unconstrained)[1]

    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_modsoss_fd - grad_modsoss_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_modsoss_fd - grad_modsoss_zy)) ≈ 0 atol = _TOL
end
=#
############################################################################################
# Test Gradients of Model with custom function
modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
data1 = randn(N)
data2 = rand(Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0]))), N)
data3 = rand(Distributions.Categorical(3), N)
_idx = rand(1:2, N)
objectiveExample = Objective(modelExample, (data1, data2, data3, _idx))

function (objective::Objective{<:ModelWrapper{ExampleModel}})(θ::NamedTuple)
    data1 = objective.data[1]
    data2 = objective.data[2]
    data3 = objective.data[3]
    σ2 = Symmetric(Diagonal(θ.σ2) * θ.ρ2 * Diagonal(θ.σ2))
    σ4 = [
        Symmetric(Diagonal(θ.σ4[iter]) * θ.ρ4[iter] * Diagonal(θ.σ4[iter])) for
        iter in eachindex(θ.σ4)
    ]
    _dist1 = Distributions.Normal(θ.μ1, θ.σ1)
    _dist2 = Distributions.MvNormal(θ.μ2, σ2)
    _dist3 = [Distributions.Normal(θ.μ3[iter], θ.σ3[iter]) for iter in eachindex(θ.μ3)]
    _dist4 = [Distributions.MvNormal(θ.μ4[iter], σ4[iter]) for iter in eachindex(θ.μ3)]
    _dist5 = Distributions.Categorical(θ.p)

    ll = sum(Distributions.logpdf(_dist1, data1[iter]) for iter in eachindex(data1))
    ll2 = sum(
        Distributions.logpdf(_dist2, @view(data2[:, iter])) for iter in size(data2, 2)
    )

    ll3 = sum(
        Distributions.logpdf(_dist3[_idx[iter]], data1[iter]) for iter in eachindex(data1)
    )
    ll4 = sum(
        Distributions.logpdf(_dist4[_idx[iter]], @view(data2[:, iter])) for
        iter in size(data2, 2)
    )

    ll5 = sum(Distributions.logpdf(_dist5, data3[iter]) for iter in eachindex(data3))

    return ll + ll2 + ll3 + ll4 + ll5
end

@testset "Objective - Log Objective AutoDiff compatibility - Vectorized Model" begin
    length(objectiveExample)
    ModelWrappers.paramnames(objectiveExample)
    theta_unconstrained = randn(length(modelExample))
    Objective(objectiveExample.model, objectiveExample.data, objectiveExample.tagged, objectiveExample.temperature)
    Objective(objectiveExample.model, objectiveExample.data, objectiveExample.tagged)
    Objective(objectiveExample.model, objectiveExample.data, keys(objectiveExample.tagged.parameter)[1:2])
    Objective(objectiveExample.model, objectiveExample.data, keys(objectiveExample.tagged.parameter)[1])
    Objective(objectiveExample.model, objectiveExample.data)

    predict(_RNG, objectiveExample)
    generate(_RNG, objectiveExample)
    generate(_RNG, objectiveExample, ModelWrappers.UpdateTrue())
    generate(_RNG, objectiveExample, ModelWrappers.UpdateFalse())
    dynamics(objectiveExample)

    @test abs(
        (log_prior(modelExample) + log_abs_det_jac(modelExample)) -
        log_prior_with_transform(modelExample),
    ) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_rd = ReverseDiff.gradient(objectiveExample, theta_unconstrained)
    grad_mod_zy = Zygote.gradient(objectiveExample, theta_unconstrained)[1]

    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

end

############################################################################################
