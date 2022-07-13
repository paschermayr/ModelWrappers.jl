############################################################################################
#=
!NOTE 1: flatten functions taken and adjusted from excellent package ParameterHandling.jl, see:
    https://github.com/invenia/ParameterHandling.jl/blob/8e998949e7fcf24d5c3f8bed5018ec300542151b/src/flatten.jl#LL1-L17
!NOTE 2: Functions redefined from discussion in:
    https://github.com/invenia/ParameterHandling.jl/issues/27
=#

############################################################################################
#!NOTE: I am very open to change type/struct names in this section :)

"""
$(TYPEDEF)

Supertype for dispatching different types of flatten. Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).

# Fields
$(TYPEDFIELDS)
"""
abstract type FlattenTypes end
struct FlattenAll <: FlattenTypes end
struct FlattenContinuous <: FlattenTypes end

"""
$(TYPEDEF)

Determines if unflatten returns original type or if type may change (AD friendly).

# Fields
$(TYPEDFIELDS)
"""
abstract type UnflattenTypes end
struct UnflattenStrict <: UnflattenTypes end
struct UnflattenFlexible <: UnflattenTypes end

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
"""
    $(FUNCTIONNAME)(x )
Construct a flatten function for 'x' given specifications in 'df'.

# Examples
```julia
```
"""
function construct_flatten end
construct_flatten(x) = construct_flatten(FlattenDefault(), UnflattenStrict(), x)
construct_flatten(constraint, x) = construct_flatten(FlattenDefault(), UnflattenStrict(), constraint, x)

construct_flatten(df::F, x) where {F<:FlattenDefault} = construct_flatten(df.output, df.flattentype, UnflattenStrict(), x)
construct_flatten(df::F, constrained, x) where {F<:FlattenDefault} = construct_flatten(df.output, df.flattentype, UnflattenStrict(), constrained, x)
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
"""
    $(FUNCTIONNAME)(x )
Convert 'x' into a Vector.

# Examples
```julia
```
"""
function flatten end
function flatten(constructor::ReConstructor, x)
    return constructor.flatten.strict(x)
end
function flatten(x)
    constructor = ReConstructor(x)
    return flatten(constructor, x), constructor
end

"""
    $(FUNCTIONNAME)(x )
Convert 'x' into a Vector that is AD compatible.

# Examples
```julia
```
"""
function flattenAD end
function flattenAD(constructor::ReConstructor, x)
    return constructor.flatten.flexible(x)
end
function flattenAD(x)
    constructor = ReConstructor(x)
    return flattenAD(constructor, x), constructor
end

"""
    $(FUNCTIONNAME)(x )
Unflatten 'x' into original shape.

# Examples
```julia
```
"""
function unflatten end
function unflatten(constructor::ReConstructor, x)
    return constructor.unflatten.strict(x)
end

"""
    $(FUNCTIONNAME)(x )
Unflatten 'x' into original shape but keep type information of 'x' for AD compatibility.

# Examples
```julia
```
"""
function unflattenAD end
function unflattenAD(constructor::ReConstructor, x)
    return constructor.unflatten.flexible(x)
end


############################################################################################
"""
    $(FUNCTIONNAME)(x )
Contains information to constrain and unconstrain parameter.

# Examples
```julia
```
"""
struct TransformConstructor{S, T}
    constrain::S
    unconstrain::T
    function TransformConstructor(constraint, x)
        #!NOTE: Transform is used to unconstrain, and inverse-transform to constrain parameter back.
        transform, inverse_transform = construct_transform(constraint, x)
        return new{typeof(inverse_transform), typeof(transform)}(inverse_transform, transform)
    end
end

function constrain(transform::T, val) where {T<:TransformConstructor}
    return constrain(transform.constrain, val)
end
function unconstrain(transform::T, val) where {T<:TransformConstructor}
    return unconstrain(transform.unconstrain, val)
end

############################################################################################
# Export
export FlattenTypes,
    FlattenAll,
    FlattenContinuous,
    UnflattenTypes,
    UnflattenStrict,
    UnflattenFlexible,
    FlattenDefault,
    FlattenConstructor,
    UnflattenConstructor,
    ReConstructor,
    flatten,
    flattenAD,
    unflatten,
    unflattenAD,
    TransformConstructor
