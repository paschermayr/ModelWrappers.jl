############################################################################################
# Include files

"""
$(FUNCTIONNAME)(x )
Construct a flatten function for 'x' given specifications in 'df'.

# Examples
```julia
```
"""
function construct_flatten end

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
# Abstract supertypes for flatten/unflatten parameter

"""
    $(FUNCTIONNAME)(x )
Convert 'x' into a Vector.

# Examples
```julia
```
"""
function flatten end

"""
    $(FUNCTIONNAME)(x )
Convert 'x' into a Vector that is AD compatible.

# Examples
```julia
```
"""
function flattenAD end

"""
    $(FUNCTIONNAME)(x )
Unflatten 'x' into original shape.

# Examples
```julia
```
"""
function unflatten end

"""
    $(FUNCTIONNAME)(x )
Unflatten 'x' into original shape but keep type information of 'x' for AD compatibility.

# Examples
```julia
```
"""
function unflattenAD end

############################################################################################
# Abstract supertypes for constrain/unconstrain
"""
$(TYPEDEF)
Abstract super type for parameter constraints.
"""
abstract type AbstractConstraint end

"Constrain `val` with given `constraint`"
function constrain end

"Unconstrain `val` with given `constraint`"
function unconstrain end

############################################################################################
# Default Methods for unconstrain_flatten and unflatten_constrain
function unconstrain_flatten end
function unconstrain_flattenAD end

function unflatten_constrain end
function unflattenAD_constrain end

############################################################################################
include("flatten/flatten.jl")
include("constrain/constrain.jl")

include("utility.jl")
include("checks.jl")

include("parameterinfo.jl")

############################################################################################
# Export
export FlattenTypes,
    FlattenAll,
    FlattenContinuous,

    UnflattenTypes,
    UnflattenStrict,
    UnflattenFlexible,

    flatten,
    flattenAD,

    unflatten,
    unflattenAD,

    AbstractConstraint,
    constrain,
    unconstrain,

    unconstrain_flatten,
    unconstrain_flattenAD,
    unflatten_constrain,
    unflattenAD_constrain
