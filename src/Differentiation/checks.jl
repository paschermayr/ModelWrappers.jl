############################################################################################
"""
$(SIGNATURES)
Check gradient computations of different backends against 'objective'.

# Examples
```julia
```

"""
function check_gradients(
    _rng::Random.AbstractRNG,
    objective::Objective,
    ADlibraries = [:ForwardDiff, :ReverseDiff, :Zygote],
    θᵤ = randn(_rng, length(objective)),
    difftune = map(backend -> DiffObjective(objective, AutomaticDiffTune(backend, objective)), ADlibraries)
)
## Compute Gradients
    ℓobjectiveresults = ℓGradientResult[]
    for iter in eachindex(difftune)
        push!(ℓobjectiveresults, log_density_and_gradient(difftune[iter], θᵤ))
        println(ADlibraries[iter], " gradient call succesfull.")
    end
## Check differences
    ℓobjective_diff = map(
        iter -> ℓobjectiveresults[1].ℓθᵤ - ℓobjectiveresults[iter].ℓθᵤ, eachindex(ℓobjectiveresults)
    )
    for iter in eachindex(ℓobjectiveresults)
        println("Log objective result difference of ", ADlibraries[1], " against ", ADlibraries[iter], ": ", ℓobjective_diff[iter])
    end

    ℓobjective_gradient_diff = map(
        iter -> sum(abs.(ℓobjectiveresults[1].∇ℓθᵤ .- ℓobjectiveresults[iter].∇ℓθᵤ)), eachindex(ℓobjectiveresults)
    )
    for iter in eachindex(ℓobjective_gradient_diff)
        println("Log objective gradient difference of ", ADlibraries[1], " against ", ADlibraries[iter], ": ", ℓobjective_gradient_diff[iter])
    end
## Compare against base Forward and ReverseDiff
    grad_fd = ForwardDiff.gradient(objective, θᵤ)
    grad_rd = ReverseDiff.gradient(objective, θᵤ)
    fdrd_diff = map(result ->
        (sum(abs.(result.∇ℓθᵤ .- grad_fd)), sum(abs.(result.∇ℓθᵤ .- grad_rd))), ℓobjectiveresults
    )
    for iter in eachindex(fdrd_diff)
        println("Log objective gradient difference of ", ADlibraries[iter], " against Forward/Reverse call: ", fdrd_diff[iter])
    end
## Return differences
    return (
        names = ADlibraries,
        difftune = difftune,
        ℓobjectiveresults = ℓobjectiveresults,
        ℓobjective_diff = ℓobjective_diff,
        ℓobjective_gradient_diff = ℓobjective_gradient_diff,
        Forward_Reverse_diff = fdrd_diff,
    )
end

############################################################################################
# Export
export check_gradients
