############################################################################################
#=
!NOTE 1: flatten functions taken and adjusted from excellent package ParameterHandling.jl, see:
    https://github.com/invenia/ParameterHandling.jl/blob/8e998949e7fcf24d5c3f8bed5018ec300542151b/src/flatten.jl#LL1-L17
!NOTE 2: Functions redefined from discussion in:
    https://github.com/invenia/ParameterHandling.jl/issues/27
!NOTE 3: This is optimized for the case where 'unflatten' is called most of the time (not flatten).
=#

############################################################################################
#!NOTE: I am very open to change type/struct names in this section :)

"Supertype for dispatching different types of flatten. Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous)."
abstract type FlattenTypes end
struct FlattenAll <: FlattenTypes end
struct FlattenContinuous <: FlattenTypes end

"Determines if unflatten returns original type or if type may change (AD friendly)."
abstract type UnflattenTypes end
struct UnflattenStrict <: UnflattenTypes end
struct UnflattenAD <: UnflattenTypes end

############################################################################################
"""
$(TYPEDEF)

Default arguments for flatten function.

# Fields
$(TYPEDFIELDS)
"""
struct FlattenDefault{T<:AbstractFloat,F<:FlattenTypes,S<:UnflattenTypes}
    #!TODO: need to find a way to make this isbits(flattendefault) == true
    "Type of flatten output"
    output::Type{T}
    "Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous)."
    flattentype::F
    "Determines if unflatten returns original type or if type may change (AD friendly)."
    unflattentype::S
    function FlattenDefault(
        output::Type{T}=Float64,
        flattentype::F=FlattenContinuous(),
        unflattentype::S=UnflattenStrict(),
    ) where {T,F<:FlattenTypes,S<:UnflattenTypes}
        return new{T,typeof(flattentype),typeof(unflattentype)}(
            output, flattentype, unflattentype
        )
    end
end
function FlattenDefault(;
    output::Type{T}=Float64,
    flattentype::F=FlattenContinuous(),
    unflattentype::S=UnflattenStrict(),
) where {T,F<:FlattenTypes,S<:UnflattenTypes}
    return FlattenDefault(output, flattentype, unflattentype)
end

############################################################################################
"""
    $(FUNCTIONNAME)(x )
Convert 'x' into a Vector.

# Examples
```julia
```
"""
function flatten end
function flatten(df::F, x) where {F<:FlattenDefault}
    return flatten(df.output, df.flattentype, df.unflattentype, x)
end
flatten(x) = flatten(FlattenDefault(), x)

################################################
function flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenStrict,
    x::Union{I,Array{I}},
) where {T<:AbstractFloat,U<:UnflattenTypes,I<:Integer}
    v = I[]
    unflatten_Integer(v) = x
    return v, unflatten_Integer
end
function flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenAD,
    x::Union{I,Array{I}},
) where {T<:AbstractFloat,U<:UnflattenTypes,I<:Integer}
    v = I[]
    unflatten_Integer(v) = x
    return v, unflatten_Integer
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenStrict, x::R
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    v = T[x]
    unflatten_to_Real(v::Union{<:Real,AbstractVector{<:Real}}) = convert(R, only(v))
    return v, unflatten_to_Real
end
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenAD, x::R
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    v = T[x]
    unflatten_to_Real_AD(v::Union{<:Real,AbstractVector{<:Real}}) = only(v)
    return v, unflatten_to_Real_AD
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenStrict, x::AbstractVector{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    buffer = zeros(R, size(x))
    unflatten_to_Vec(v::Union{<:Real,AbstractVector{<:Real}}) = fill_array!(buffer, v)
    return Vector{T}(x), unflatten_to_Vec
end
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenAD, x::AbstractVector{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    return Vector{T}(x), identity
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenStrict, x::AbstractArray{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    x_vec, from_vec = flatten(T, flattentype, unflattentype, vec(x))
    buffer = zeros(R, size(x))
    unflatten_to_Array(v::Union{<:Real,AbstractVector{<:Real}}) = fill_array!(buffer, v)
    return x_vec, unflatten_to_Array
end
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenAD, x::AbstractArray{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    x_vec, from_vec = flatten(T, flattentype, unflattentype, vec(x))
    function unflatten_to_Array_AD(v::Union{<:Real,AbstractVector{<:Real}})
        return fill_array!(zeros(eltype(v), size(x)), v)
    end
    return x_vec, unflatten_to_Array_AD
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x::AbstractArray
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    x_vecs_and_backs = map(x) do xᵢ
        flatten(T, flattentype, unflattentype, xᵢ)
    end
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    sz = cumsum(map(length, x_vecs))
    function unflatten_to_AbstractArray(x_vec::Union{<:Real,AbstractVector{<:Real}})
        x_Vec = [
            backs[n](view(x_vec, (sz[n] - length(x_vecs[n]) + 1):sz[n])) for
            n in eachindex(x)
        ]
        return x_Vec
    end
    return reduce(vcat, x_vecs), unflatten_to_AbstractArray
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x::Tuple
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    x_vecs_and_backs = map(x) do xᵢ
        flatten(T, flattentype, unflattentype, xᵢ)
    end
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = cumsum(lengths)
    function unflatten_to_Tuple(v::Union{<:Real,AbstractVector{<:Real}})
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(view(v, (s - l + 1):s))
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenStrict, x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes,names}
    x_vec, unflatten = flatten(T, flattentype, unflattentype, values(x))
    function unflatten_to_NamedTuple(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec_vec = unflatten(v)
        return typeof(x)(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end
function flatten(
    output::Type{T}, flattentype::F, unflattentype::UnflattenAD, x::NamedTuple{names}
) where {T<:AbstractFloat,F<:FlattenTypes,names}
    x_vec, unflatten = flatten(T, flattentype, unflattentype, values(x))
    function unflatten_to_NamedTuple_AD(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec_vec = unflatten(v)
        #!NOTE: cannot use typeof(x) in AD as Floats are converted to Duals
        return NamedTuple{names}(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple_AD
end
# Convenient constructor for subset of x
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x::NamedTuple, sym::S
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    S<:Union{NamedTuple,NTuple{N,Symbol} where {N}},
}
    return flatten(T, flattentype, unflattentype, subset(x, sym))
end

############################################################################################
# Add methods for flattening different constraints
function flatten(df::F, x, constraint) where {F<:FlattenDefault}
    return flatten(df.output, df.flattentype, df.unflattentype, x, constraint)
end
flatten(x, constraint) = flatten(FlattenDefault(), x, constraint)

################################################
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::AbstractArray,
    constraint::AbstractArray,
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    x_vecs_and_backs = map(x, constraint) do xᵢ, constraintᵢ
        flatten(T, flattentype, unflattentype, xᵢ, constraintᵢ)
    end
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    sz = cumsum(map(length, x_vecs))
    function unflatten_to_AbstractArray(x_vec::Union{<:Real,AbstractVector{<:Real}})
        x_Vec = [
            backs[n](view(x_vec, (sz[n] - length(x_vecs[n]) + 1):sz[n])) for
            n in eachindex(x)
        ]
        return x_Vec
    end
    return reduce(vcat, x_vecs), unflatten_to_AbstractArray
end

################################################
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x::Tuple, constraint::Tuple
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    x_vecs_and_backs = map(x, constraint) do xᵢ, constraintᵢ
        flatten(T, flattentype, unflattentype, xᵢ, constraintᵢ)
    end
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = cumsum(lengths)
    function unflatten_to_Tuple(v::Union{<:Real,AbstractVector{<:Real}})
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(view(v, (s - l + 1):s))
        end
    end
    return reduce(vcat, x_vecs), unflatten_to_Tuple
end

################################################
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::NamedTuple{names},
    constraint::NamedTuple,
) where {T<:AbstractFloat,F<:FlattenTypes,names}
    x_vec, unflatten = flatten(T, flattentype, unflattentype, values(x), values(constraint))
    function unflatten_to_NamedTuple(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec_vec = unflatten(v)
        return typeof(x)(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenAD,
    x::NamedTuple{names},
    constraint::NamedTuple,
) where {T<:AbstractFloat,F<:FlattenTypes,names}
    x_vec, unflatten = flatten(T, flattentype, unflattentype, values(x), values(constraint))
    function unflatten_to_NamedTuple_AD(v::Union{<:Real,AbstractVector{<:Real}})
        v_vec_vec = unflatten(v)
        #!NOTE: Cannot use typeof(x) for AD case as Floats are converted to Duals
        return NamedTuple{names}(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple_AD
end
# Convenient constructor for subset of x
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::NamedTuple,
    constraint::NamedTuple,
    sym::S,
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    S<:Union{NamedTuple,NTuple{N,Symbol} where {N}},
}
    return flatten(T, flattentype, unflattentype, subset(x, sym), subset(constraint, sym))
end

############################################################################################
# Export
export FlattenTypes,
    FlattenAll,
    FlattenContinuous,
    UnflattenTypes,
    UnflattenStrict,
    UnflattenAD,
    FlattenDefault,
    flatten
