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
        @test length(_model_temp) == length(_target)
        @test abs(log_prior(_model_temp, _target) - log_prior(_model_temp)) <= _TOL
        @test abs(log_abs_det_jac(_model_temp, _target) - log_abs_det_jac(_model_temp)) <=
              _TOL
        ## Type Check - Unflatten/Constrain
        theta_unconstrained_vec = randn(length(_target))
        theta_constrained = unflatten_constrain(_model_temp, theta_unconstrained_vec)
        theta_constrained2 = unflatten_constrain(
            _model_temp, _target, theta_unconstrained_vec
        )
        @test typeof(theta_constrained) == typeof(_model_temp.val)
        @test typeof(theta_constrained2) == typeof(_model_temp.val)
        _θ1  = flatten(_model_temp.info.reconstruct, theta_constrained)
        _θ2  = flatten(_target.info.reconstruct, theta_constrained2)
        @test sum(abs.(_θ1 - _θ2)) ≈ 0 atol = _TOL
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
        ModelWrappers.length(_target)
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
