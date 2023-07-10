############################################################################################
#=
!NOTE 1: flatten functions taken and adjusted from excellent package ParameterHandling.jl, see:
    https://github.com/invenia/ParameterHandling.jl/blob/8e998949e7fcf24d5c3f8bed5018ec300542151b/src/flatten.jl#LL1-L17
!NOTE 2: Functions redefined from discussion in:
    https://github.com/invenia/ParameterHandling.jl/issues/27
=#

############################################################################################
"""
$(TYPEDEF)

Default arguments for flatten function.

# Fields
$(TYPEDFIELDS)
"""
struct FlattenDefault{T<:AbstractFloat,F<:FlattenTypes}
    "Type of flatten output"
    output::Type{T}
    "Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous)."
    flattentype::F
end
function FlattenDefault(;
    output::Type{T}=Float64,
    flattentype::F=FlattenContinuous(),
) where {T,F<:FlattenTypes}
    return FlattenDefault(output, flattentype)
end

############################################################################################
construct_flatten(x) = construct_flatten(FlattenDefault(), UnflattenStrict(), x)
construct_flatten(constraint, x) = construct_flatten(FlattenDefault(), UnflattenStrict(), constraint, x)

construct_flatten(df::F, x) where {F<:FlattenDefault} = construct_flatten(df.output, df.flattentype, UnflattenStrict(), x)
construct_flatten(df::F, constraint, x) where {F<:FlattenDefault} = construct_flatten(df.output, df.flattentype, UnflattenStrict(), constraint, x)
construct_flatten(df::F, unflattentype::U, x) where {F<:FlattenDefault, U<:UnflattenTypes} = construct_flatten(df.output, df.flattentype, unflattentype, x)
construct_flatten(df::F, unflattentype::U, constraint, x) where {F<:FlattenDefault, U<:UnflattenTypes} = construct_flatten(df.output, df.flattentype, unflattentype, constraint, x)

############################################################################################
"""
$(TYPEDEF)

Contains information for flatten construct.

# Fields
$(TYPEDFIELDS)
"""
struct FlattenConstructor{S<:Function, T<:Function}
    strict::S
    flexible::T
    function FlattenConstructor(strict::S, flexible::T) where {S<:Function, T<:Function}
        return new{S,T}(strict, flexible)
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains information for unflatten construct.

# Fields
$(TYPEDFIELDS)
"""
struct UnflattenConstructor{S<:Function, T<:Function}
    strict::S
    flexible::T
    function UnflattenConstructor(strict::S, flexible::T) where {S<:Function, T<:Function}
        return new{S,T}(strict, flexible)
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains information for flatten/unflatten construct.

# Fields
$(TYPEDFIELDS)
"""
struct ReConstructor{F<:FlattenDefault, S<:FlattenConstructor, T<:UnflattenConstructor}
    default::F
    flatten::S
    unflatten::T
    function ReConstructor(default::F, flatten::S, unflatten::T) where {F<:FlattenDefault, S<:FlattenConstructor, T<:UnflattenConstructor}
        return new{F,S,T}(default, flatten, unflatten)
    end
end
function ReConstructor(flattendefault::FlattenDefault, x)
# Assign flatten constructors
    flatten, unflatten = construct_flatten(flattendefault, UnflattenStrict(), x)
    flattenAD, unflattenAD = construct_flatten(flattendefault, UnflattenFlexible(), x)
    flatten_constructor = FlattenConstructor(flatten, flattenAD)
    unflatten_constructor = UnflattenConstructor(unflatten, unflattenAD)
# Return structs
    return ReConstructor(flattendefault, flatten_constructor, unflatten_constructor)
end
function ReConstructor(flattendefault::FlattenDefault, constraint, x)
# Assign flatten constructors
    flatten, unflatten = construct_flatten(flattendefault, UnflattenStrict(), constraint, x)
    flattenAD, unflattenAD = construct_flatten(flattendefault, UnflattenFlexible(), constraint, x)
    flatten_constructor = FlattenConstructor(flatten, flattenAD)
    unflatten_constructor = UnflattenConstructor(unflatten, unflattenAD)
# Return structs
    return ReConstructor(flattendefault, flatten_constructor, unflatten_constructor)
end
function ReConstructor(x)
    return ReConstructor(FlattenDefault(), x)
end
function ReConstructor(constraint, x)
    return ReConstructor(FlattenDefault(), constraint, x)
end

############################################################################################
function flatten(constructor::ReConstructor, x)
    return constructor.flatten.strict(x)
end
function flatten(x)
    constructor = ReConstructor(x)
    return flatten(constructor, x), constructor
end

function flattenAD(constructor::ReConstructor, x)
    return constructor.flatten.flexible(x)
end
function flattenAD(x)
    constructor = ReConstructor(x)
    return flattenAD(constructor, x), constructor
end

function unflatten(constructor::ReConstructor, x)
    return constructor.unflatten.strict(x)
end
function unflattenAD(constructor::ReConstructor, x)
    return constructor.unflatten.flexible(x)
end

############################################################################################
# Export
export FlattenDefault,
    FlattenConstructor,
    UnflattenConstructor,
    ReConstructor,
    flatten,
    flattenAD,
    unflatten,
    unflattenAD
