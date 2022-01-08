################################################################################
# Assign Model and Target Parameter
_modelProb = ModelWrapper(ProbModel(), val_dist)
_syms = [
    keys(_modelProb.val),
    keys(_modelProb.val)[1],
    keys(_modelProb.val)[end],
    (:d_a1, :d_a2, :d_a3, :d_a5, :d_a6, :d_a7, :d_a8, :d_b1, :d_b2),
]
_targets = [Tagged(_modelProb, _syms[iter]) for iter in eachindex(_syms)]
_params = [sample(_modelProb, _targets[iter]) for iter in eachindex(_syms)]

@testset "Tagged - Model parameter" begin
    for iter in eachindex(_syms)
        ## Assign Sub Model
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
        _θ1, _ = flatten(theta_constrained, _model_temp.info.constraint)
        _θ2, _ = flatten(theta_constrained2, _target.info.constraint)
        @test sum(abs.(_θ1 - _θ2)) ≈ 0 atol = _TOL
    end
end
