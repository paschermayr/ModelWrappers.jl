############################################################################################
"""
$(SIGNATURES)
Check if 'θ' is of finite value and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkfinite(θ::T, max_val::R=max_val) where {T<:Real,R<:Real}
    return isfinite(θ) && θ < max_val
end
function _checkfinite(θ::Array{T}, max_val::R=max_val) where {T<:Real,R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !_checkfinite(θ[iter], max_val)
            return false
        end
    end
    return true
end
function _checkfinite(θ::AbstractArray, max_val::R=max_val) where {R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !_checkfinite(θ[iter], max_val)
            return false
        end
    end
    return true
end
function _checkfinite(θ::Tuple, max_val::R=max_val) where {R<:Real}
    return all(map(_checkfinite, θ))
end
function _checkfinite(θ::N, max_val::R=max_val) where {N<:NamedTuple} where {R<:Real}
    return all(map(_checkfinite, values(θ)))
end

############################################################################################
"""
$(SIGNATURES)
Check if 'prior' is a valid density and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkprior(prior)
    return false
end
function _checkprior(prior::S) where {S<:Distributions.Distribution}
    return true
end
function _checkprior(prior::AbstractVector{T}) where {T}
    @inbounds @simd for iter in eachindex(prior)
        if !_checkprior(prior[iter])
            return false
        end
    end
    return true
end
function _checkprior(prior::AbstractArray)
    @inbounds @simd for iter in eachindex(prior)
        if !_checkprior(prior[iter])
            return false
        end
    end
    return true
end
function _checkprior(prior::N) where {N<:NamedTuple}
    return all(map(_checkprior, prior))
end

############################################################################################
"""
$(SIGNATURES)
Check if argument is not fixed. Returns NamedTuple with true/false. Needed in addition to _checkprior for nested NamedTuples. Not exported.

# Examples
```julia
```

"""
function _checksampleable(constraint)
    return false
end
function _checksampleable(constraint::S) where {S<:Distributions.Distribution}
    return true
end
function _checksampleable(constraint::AbstractVector{T}) where {T}
    @inbounds @simd for iter in eachindex(constraint)
        if !_checksampleable(constraint[iter])
            return false
        end
    end
    return true
end
function _checksampleable(constraint::AbstractArray)
    @inbounds @simd for iter in eachindex(constraint)
        if !_checksampleable(constraint[iter])
            return false
        end
    end
    return true
end
function _checksampleable(constraint::NamedTuple{names}) where {names}
    return NamedTuple{names}(map(_checksampleable, constraint))
end

############################################################################################
#=
"""
$(SIGNATURES)
Check if provided val-constraint combination is valid for Param struct. Not exported.

# Examples
```julia
```

"""
function _checkparam(
    _rng::Random.AbstractRNG,
    constraint::AbstractArray,
    val::AbstractArray,
)
    return all(
        map(constraint, val) do constraintᵢ, valᵢ
            _checkparam(_rng, constraintᵢ, valᵢ)
        end,
    )
end
_checkparam(constraint, val) = _checkparam(Random.GLOBAL_RNG, constraint, val)
=#
############################################################################################
"""
$(SIGNATURES)
Check if all values in (Nested) NamedTuple are a 'Param' struct and return Bool. Not exported.

# Examples
```julia
```

"""
function _checkparams(param)
    return false
end
function _checkparams(param::Param)
    return true
end

function _checkparams(param::N) where {N<:NamedTuple}
    return all(map(_checkparams, param))
end

############################################################################################
# Export
