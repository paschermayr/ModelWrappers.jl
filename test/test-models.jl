############################################################################################
# Basic Functionality
_modelProb = ModelWrapper(ProbModel(), val_dist)
@testset "Models - basic functionality" begin
    
    @test length_constrained(_modelProb) == length_constrained(_modelProb.info)
    @test length_unconstrained(_modelProb) == length_unconstrained(_modelProb.info)
    
    ## Type Check 1 - Constrain/Unconstrain
    length_constrained(_modelProb)
    length_unconstrained(_modelProb)
    theta_unconstrained_vec = randn(length_unconstrained(_modelProb))
    
    
    val_flat = flatten(_modelProb)
    val_unflat = unflatten(_modelProb, val_flat)
    @test length(val_unflat) == length(_modelProb.val)

    val_unconstrained = unconstrain(_modelProb)
    val_constrained = constrain(_modelProb, val_unconstrained)

    val_flat_unconstrained = unconstrain_flatten(_modelProb)
    val_unflat_constrained = unflatten_constrain(_modelProb, val_flat_unconstrained)
    @test length(val_unflat_constrained) == length(_modelProb.val)
    unflattenAD_constrain(_modelProb, val_flat_unconstrained)

    ## Check if densities match
    @test log_prior(_modelProb) + log_abs_det_jac(_modelProb) ≈
          log_prior_with_transform(_modelProb)
    ## Utility functions
    ModelWrappers.generate_showvalues(_modelProb)()
    unconstrain(_modelProb)
    flatten(_modelProb)
    unconstrain_flatten(_modelProb)
    unconstrain_flattenAD(_modelProb)

    fill(_modelProb, _modelProb.val)
    fill!(_modelProb, _modelProb.val)
    subset(_modelProb, keys(_modelProb.val))
    unflatten!(_modelProb, flatten(_modelProb))
    unflatten_constrain!(_modelProb, unconstrain_flatten(_modelProb))
    sample(_RNG, _modelProb)
    sample(_modelProb)
    sample!(_RNG, _modelProb)
    sample!(_modelProb)

    #Explore Initialization patterns:
    PriorInitialization()
    OptimInitialization()

end

############################################################################################
## Model with transforms in lower dimensions
_modelExample = ModelWrapper(ExampleModel(), _val_examplemodel)
_tagged = Tagged(_modelExample)
@testset "Models - Model with transforms in lower dimensions" begin
    ## Model Length accounting discrete parameter
    length_constrained(_modelExample)
    length_unconstrained(_modelExample)
    theta_unconstrained_vec = randn(length_unconstrained(_modelExample))
    
    
    val_flat = flatten(_modelExample)
    val_unflat = unflatten(_modelExample, val_flat)
    @test length(val_unflat) == length(_modelExample.val)

    val_unconstrained = unconstrain(_modelExample)
    val_constrained = constrain(_modelExample.info, val_unconstrained)

    val_flat_unconstrained = unconstrain_flatten(_modelExample)
    val_unflat_constrained = unflatten_constrain(_modelExample, val_flat_unconstrained)
    @test length(val_unflat_constrained) == length(_modelExample.val)

    
    ## Check if densities match
    @test log_prior(_modelExample) + log_abs_det_jac(_modelExample) ≈
          log_prior_with_transform(_modelExample)
    ## Check utility functions
    @test length_unconstrained(_modelExample) == 23
    @test ModelWrappers.paramnames(_modelExample) == keys(_val_examplemodel)
    fill(_modelExample, _tagged, _modelExample.val)
    fill!(_modelExample, _tagged, _modelExample.val)
end

############################################################################################
struct NonBijectModel <: ModelName end
val_nonbjiject = (
    a=Param(LKJ(3,1), rand(LKJ(3,1))),
    b=Param(LKJCholesky(3,1), rand(LKJCholesky(3,1))), 
    c=Param(InverseWishart(10, [3. .1 ; .1 2.]), rand(InverseWishart(10, [3. .1 ; .1 2.]))),
    d=Param(Dirichlet(3,3), [.1, .2, .7]), 
)
_modelExample = ModelWrapper(NonBijectModel(), val_nonbjiject)
_tagged = Tagged(_modelExample)
@testset "Models - Model with transforms in lower dimensions" begin
    ## Model Length accounting discrete parameter
    length_constrained(_modelExample)
    length_unconstrained(_modelExample)
    theta_unconstrained_vec = randn(length_unconstrained(_modelExample))
    
    val_flat = flatten(_modelExample)
    val_unflat = unflatten(_modelExample, val_flat)
    @test length(val_unflat) == length(_modelExample.val)

    val_unconstrained = unconstrain(_modelExample)
    val_constrained = constrain(_modelExample, val_unconstrained)

    val_flat_unconstrained = unconstrain_flatten(_modelExample)
    val_unflat_constrained = unflatten_constrain(_modelExample, val_flat_unconstrained)
    @test length(val_unflat_constrained) == length(_modelExample.val)

    
    ## Check if densities match
    @test log_prior(_modelExample) + log_abs_det_jac(_modelExample) ≈
          log_prior_with_transform(_modelExample)
    ## Check utility functions
    @test length_constrained(_modelExample) == 25
    @test length_unconstrained(_modelExample) == 11
    fill(_modelExample, _tagged, _modelExample.val)
    fill!(_modelExample, _tagged, _modelExample.val)
end