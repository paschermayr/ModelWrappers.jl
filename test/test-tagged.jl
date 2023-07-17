################################################################################
# Assign Model and Target Parameter
_modelProb = ModelWrapper(ProbModel(), val_dist)
_syms = [
    keys(_modelProb.val),
    keys(_modelProb.val)[1],
    keys(_modelProb.val)[end],
    (:d_a1, :d_a2, :d_a3, :d_a5, :d_a6, :d_a7, :d_a8),
]
_targets = [Tagged(_modelProb, _syms[iter]) for iter in eachindex(_syms)]
_params = [sample(_modelProb, _targets[iter]) for iter in eachindex(_syms)]

@testset "Tagged - Model parameter" begin
        ## Assign Sub Model
        for iter in eachindex(_syms)
        _sym = _syms[iter]
        _target = _targets[iter]
        _param = _params[iter]
        _model_temp = ModelWrapper(subset(val_dist, _sym))
        ## Compute logdensities and check length
        @test length_constrained(_model_temp) == length_constrained(_target)
        @test length_unconstrained(_model_temp) == length_unconstrained(_target)
        
        @test abs(log_prior(_model_temp, _target) - log_prior(_model_temp)) <= _TOL
        @test abs(log_abs_det_jac(_model_temp, _target) - log_abs_det_jac(_model_temp)) <=
              _TOL
        ## Type Check - Unflatten/Constrain
        val_flat = flatten(_model_temp, _target)
        val_unflat = unflatten(_model_temp, _target, val_flat)
#        @test length(val_unflat) == length(_model_temp.val)
    
        val_unconstrained = unconstrain(_model_temp, _target)
        constrain(_model_temp, _target, val_unconstrained)


        val_flat_unconstrained = unconstrain_flatten(_model_temp, _target)
        unconstrain_flattenAD(_model_temp, _target)
        val_unflat_constrained = unflatten_constrain(_model_temp, _target, val_flat_unconstrained)
        @test length(val_unflat_constrained) == length(_model_temp.val)

        ## Utility functions
        log_prior(_target, _model_temp.val)
        θ_flat = flatten(_model_temp, _target)
        unflatten(_model_temp, _target, θ_flat)
        unflatten!(_model_temp, _target, θ_flat)
        θ_flat_unconstrained = unconstrain_flatten(_model_temp, _target)
        unflatten_constrain!(_model_temp, _target, θ_flat_unconstrained)
        log_prior_with_transform(_model_temp, _target)

        subset(_model_temp, _target)
        ModelWrappers.generate_showvalues(_model_temp, _target)()
        ModelWrappers.paramnames(_target)
        fill(_model_temp, _target, _model_temp.val)
        fill!(_model_temp, _target, _model_temp.val)
        _model_temp.val
        sample(_RNG, _model_temp, _target)
        sample(_model_temp, _target)
        sample!(_RNG, _model_temp, _target)
        sample!(_model_temp, _target)

        print(_model_temp, _target)

    end
end
