############################################################################################
"""
$(SIGNATURES)
Transform 'θ' into an unconstrained space given 'b'. Returns same type as 'θ'.

# Examples
```julia
```

"""
function unconstrain(b::S, θ::T) where {S<:Bijectors.Bijector,T}
    return b(θ)
end
function unconstrain(bᵥ::AbstractArray, θ::AbstractArray) where {T}
    return map(unconstrain, bᵥ, θ)
end
function unconstrain(bᵥ::A, θ::B) where {A<:NamedTuple,B<:NamedTuple}
    return typeof(θ)(map(unconstrain, bᵥ, θ))
end

############################################################################################
"""
$(SIGNATURES)
Inverse-Transform 'θᵤ' into a constrained space given 'b⁻¹'. Returns same type as 'θᵤ'.

# Examples
```julia
```

"""
function constrain(b⁻¹::S, θᵤ::T) where {S<:Bijectors.Bijector,T}
    return b⁻¹(θᵤ)
end
function constrain(b⁻¹ᵥ::AbstractArray, θᵤ::AbstractArray) where {T}
    return map(constrain, b⁻¹ᵥ, θᵤ)
end
#=
#!NOTE: ReverseDiff only works with NamedTuple{names} instead of typeof(b⁻¹ᵥ). Takes much more time - might remove ReverseDiff once other ReverseModeAD package comes up.
# see: https://github.com/JuliaDiff/ReverseDiff.jl/issues/178
function constrain(b⁻¹ᵥ::A, θᵤ::B) where {A<:NamedTuple, B<:NamedTuple}
    return typeof(θᵤ)(map(constrain, b⁻¹ᵥ, θᵤ))
end
=#
function constrain(b⁻¹ᵥ::NamedTuple{names}, θₜ::B) where {names,B<:NamedTuple}
    return NamedTuple{names}(map(iter -> constrain(b⁻¹ᵥ[iter], θₜ[iter]), eachindex(θₜ)))
end

############################################################################################
"""
$(SIGNATURES)
Transform user constraint to Bijector. Not exported.

# Examples
```julia
```

"""
function _to_bijector(infoᵥ::AbstractArray)
    return map(_to_bijector, infoᵥ)
end
function _to_bijector(infoᵥ::NamedTuple{names}) where {names}
    return NamedTuple{names}(Tuple(map(_to_bijector, infoᵥ)))
end
"""
$(SIGNATURES)
Transform Bijector to its inverse. Not exported.

# Examples
```julia
```

"""
function _to_inv_bijector(info::Bijectors.Bijector)
    return Bijectors.inverse(info)
end
function _to_inv_bijector(infoᵥ::AbstractArray)
    return map(_to_inv_bijector, infoᵥ)
end
function _to_inv_bijector(infoᵥ::NamedTuple{names}) where {names}
    return NamedTuple{names}(Tuple(map(_to_inv_bijector, infoᵥ)))
end

############################################################################################
#export
export constrain, unconstrain
