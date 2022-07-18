############################################################################################
"""
$(SIGNATURES)
Use Prior predictive samples to check model assumptions. Needs dispatch on simulate(rng, model).

# Examples
```julia
```

"""
function predictive(_rng::Random.AbstractRNG, objective::Objective, init::PriorInitialization, iter::Integer)
    dataᵥ = Vector{typeof(objective.data)}(undef, iter)
    for idx in Base.OneTo(iter)
        # Sample from prior for new data points
        init(_rng, nothing, objective)
        # Simulate new data
        dataᵥ[idx] = ModelWrappers.simulate(_rng, objective.model)
    end
    return dataᵥ
end

############################################################################################
# Export
export predictive
