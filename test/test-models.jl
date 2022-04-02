############################################################################################
# Basic Functionality
_modelProb = ModelWrapper(ProbModel(), val_dist)
@testset "Models - basic functionality" begin
    ## Type Check 1 - Constrain/Unconstrain
    theta_unconstrained_vec = randn(length(_modelProb))
    theta_unconstrained = unflatten(_modelProb, theta_unconstrained_vec)
    @test typeof(theta_unconstrained) == typeof(_modelProb.val)
    theta_constrained = constrain(_modelProb.info.b⁻¹, theta_unconstrained)
    theta_constrained2 = unflatten_constrain(_modelProb, theta_unconstrained_vec)
    @test typeof(theta_constrained) == typeof(_modelProb.val)
    @test typeof(theta_constrained2) == typeof(_modelProb.val)
    ## Type Check 2 - Flatten/Unflatten
    _θ1, _ = flatten(theta_constrained, _modelProb.info.constraint)
    _θ2, _ = flatten(theta_constrained2, _modelProb.info.constraint)
    @test sum(abs.(_θ1 - _θ2)) ≈ 0 atol = _TOL
    ## Check if densities match
    @test log_prior(_modelProb) + log_abs_det_jac(_modelProb) ≈
          log_prior_with_transform(_modelProb)
    ## Utility functions
    unconstrain(_modelProb)
    flatten(_modelProb)
    unconstrain_flatten(_modelProb)
    simulate(_modelProb)
    fill(_modelProb, _modelProb.val)
    fill!(_modelProb, _modelProb.val)
    subset(_modelProb, keys(_modelProb.val))
    unflatten!(_modelProb, flatten(_modelProb))
    unflatten_constrain!(_modelProb, unconstrain_flatten(_modelProb))
    sample(_RNG, _modelProb)
    sample(_modelProb)
    sample!(_RNG, _modelProb)
    sample!(_modelProb)
end

############################################################################################
## Model with transforms in lower dimensions

_modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
_tagged = Tagged(_modelExample)
@testset "Models - Model with transforms in lower dimensions" begin
    ## Model Length accounting discrete parameter
    unconstrain(_modelExample)
    flatten(_modelExample)
    unconstrain_flatten(_modelExample)
    ## Type Check 1 - Constrain/Unconstrain
    theta_unconstrained_vec = randn(length(_modelExample))
    theta_unconstrained = unflatten(_modelExample, theta_unconstrained_vec)
    @test typeof(theta_unconstrained) == typeof(_modelExample.val)
    theta_constrained = constrain(_modelExample.info.b⁻¹, theta_unconstrained)
    theta_constrained2 = unflatten_constrain(_modelExample, theta_unconstrained_vec)
    @test typeof(theta_constrained) == typeof(_modelExample.val)
    @test typeof(theta_constrained2) == typeof(_modelExample.val)
    ## Type Check 2 - Flatten/Unflatten
    _θ1, _ = flatten(theta_constrained, _modelExample.info.constraint)
    _θ2, _ = flatten(theta_constrained2, _modelExample.info.constraint)
    @test sum(abs.(_θ1 - _θ2)) ≈ 0 atol = _TOL
    ## Check if densities match
    @test log_prior(_modelExample) + log_abs_det_jac(_modelExample) ≈
          log_prior_with_transform(_modelExample)
    ## Check utility functions
    @test length(_modelExample) == 23
    @test ModelWrappers.paramnames(_modelExample) == keys(_val_examplemodel)
    fill(_modelExample, _tagged, _modelExample.val)
    fill!(_modelExample, _tagged, _modelExample.val)
end

############################################################################################
