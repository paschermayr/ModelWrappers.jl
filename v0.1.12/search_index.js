var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be properly done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ModelWrappers","category":"page"},{"location":"#ModelWrappers","page":"Home","title":"ModelWrappers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ModelWrappers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ModelWrappers]","category":"page"},{"location":"#ModelWrappers.max_val","page":"Home","title":"ModelWrappers.max_val","text":"Maximum value after transformation before tagged as non-finite.\n\n\n\n\n\n","category":"constant"},{"location":"#ModelWrappers.min_Δ","page":"Home","title":"ModelWrappers.min_Δ","text":"Smallest decrease allowed in the log objective results before tagged as divergent.\n\n\n\n\n\n","category":"constant"},{"location":"#ModelWrappers.AbstractConstraint","page":"Home","title":"ModelWrappers.AbstractConstraint","text":"abstract type AbstractConstraint\n\nAbstract super type for parameter constraints.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.AbstractDifferentiableTune","page":"Home","title":"ModelWrappers.AbstractDifferentiableTune","text":"abstract type AbstractDifferentiableTune\n\nAbstract super type for Tuning structs of differentiable functions.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.AnalyticalDiffTune","page":"Home","title":"ModelWrappers.AnalyticalDiffTune","text":"struct AnalyticalDiffTune{G<:Function} <: ModelWrappers.AbstractDifferentiableTune\n\nStores information for evaluating and taking the gradient of an objective function.\n\nFields\n\ngradient::Function\nGradient as function of ℓobjective and parameter vector in unconstrained space, gradient(ℓobjective, θᵤ).\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.AutomaticDifferentiationMethod","page":"Home","title":"ModelWrappers.AutomaticDifferentiationMethod","text":"abstract type AutomaticDifferentiationMethod\n\nAbstract super type for Supported Automatic Differentiation backends.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.BaseModel","page":"Home","title":"ModelWrappers.BaseModel","text":"struct BaseModel <: ModelName\n\nDefault modelname of Baytes.Model struct.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Constrained","page":"Home","title":"ModelWrappers.Constrained","text":"struct Constrained{T<:Real} <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps scalar parameter constrained.\n\nFields\n\nmin::Real\nmax::Real\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.DiffObjective","page":"Home","title":"ModelWrappers.DiffObjective","text":"struct DiffObjective{O<:Objective, T<:ModelWrappers.AbstractDifferentiableTune}\n\nObjective struct with additional information about AD backend and configuration.\n\nFields\n\nobjective::Objective\nObjective as function of a parameter vector in unconstrained space.\ntune::ModelWrappers.AbstractDifferentiableTune\nAutomatic Differentiation configurations.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Fixed","page":"Home","title":"ModelWrappers.Fixed","text":"struct Fixed <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps parameter fixed. Useful for assigning buffer values for functions of parameter.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.FlattenDefault","page":"Home","title":"ModelWrappers.FlattenDefault","text":"struct FlattenDefault{T<:AbstractFloat, F<:FlattenTypes, S<:UnflattenTypes}\n\nDefault arguments for flatten function.\n\nFields\n\noutput::Type{T} where T<:AbstractFloat\nType of flatten output\nflattentype::FlattenTypes\nDetermines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).\nunflattentype::UnflattenTypes\nDetermines if unflatten returns original type or if type may change (AD friendly).\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.FlattenTypes","page":"Home","title":"ModelWrappers.FlattenTypes","text":"Supertype for dispatching different types of flatten. Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ModelName","page":"Home","title":"ModelWrappers.ModelName","text":"abstract type ModelName\n\nAbstract super type for Baytes Models.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ModelWrapper","page":"Home","title":"ModelWrappers.ModelWrapper","text":"mutable struct ModelWrapper{M<:(Union{ModelName, P} where P), A<:NamedTuple, B<:ParameterInfo} <: BaytesCore.AbstractModelWrapper\n\nBaytes Model struct.\n\nContains information about current Model value, name, and information, see also ParameterInfo.\n\nFields\n\nval::NamedTuple\nCurrent Model values as NamedTuple - works with Nested Tuples.\ninfo::ParameterInfo\nInformation about parameter distributions, transformations and constraints, see ParameterInfo.\nid::Union{ModelName, P} where P\nModel id, per default BaseModel. Useful for dispatching ModelWrapper struct.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Objective","page":"Home","title":"ModelWrappers.Objective","text":"struct Objective{M<:ModelWrapper, D, T<:Tagged, F<:AbstractFloat} <: BaytesCore.AbstractObjective\n\nFunctor to calculate 'ℓfunc' and gradient at unconstrained 'θᵤ', including eventual Jacobian adjustments.\n\nFields\n\nmodel::ModelWrapper\ndata::Any\ntagged::Tagged\ntemperature::AbstractFloat\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Objective-Tuple{Any}","page":"Home","title":"ModelWrappers.Objective","text":"Functor to call target function for Model given parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Param","page":"Home","title":"ModelWrappers.Param","text":"struct Param{A, B}\n\nUtility struct to define Parameter in a way ModelWrappers.jl can handle. Will be separated in ModelWrapper struct for type stability.\n\nFields\n\nval::Any\nconstraint::Any\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ParameterInfo","page":"Home","title":"ModelWrappers.ParameterInfo","text":"struct ParameterInfo{A, B, C, D<:FlattenDefault, E<:Function, F<:Function}\n\nContains information about parameter distributions, transformations and constraints.\n\nFields\n\nconstraint::Any\nConstraint distribution/boundaries for all model parameter.\nb::Any\nBijector for all model parameter.\nb⁻¹::Any\nInverse-Bijector for all model parameter.\nflattendefault::FlattenDefault\nDefault Flattening setting\nunflatten::Function\nFunction to unflatten model parameter, if provided as a vector.\nunflatten_AD::Function\nFunction to unflatten model parameter, if provided as a vector.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Tagged","page":"Home","title":"ModelWrappers.Tagged","text":"struct Tagged{A<:NamedTuple, B<:ParameterInfo}\n\nStores information for a subset of 'model' parameter.\n\nFields\n\nparameter::NamedTuple\nSubset of ModelWrapper parameter names.\ninfo::ParameterInfo\nInformation about subset of parameter distributions, transformations and constraints, see ParameterInfo.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.Unconstrained","page":"Home","title":"ModelWrappers.Unconstrained","text":"struct Unconstrained <: AbstractConstraint\n\nUtility struct to help assign boundaries to parameter - keeps parameter unconstrained. Useful for assigning buffer values for functions of parameter.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.UnflattenTypes","page":"Home","title":"ModelWrappers.UnflattenTypes","text":"Determines if unflatten returns original type or if type may change (AD friendly).\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ℓDensityResult","page":"Home","title":"ModelWrappers.ℓDensityResult","text":"struct ℓDensityResult{T, S} <: ℓObjectiveResult\n\nStores result for log density and parameter for 'ℓobjective' evaluation at 'parameter'.\n\nFields\n\nθᵤ::Any\nParameter in unconstrained space.\nℓθᵤ::Any\nLog density at θᵤ.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ℓGradientResult","page":"Home","title":"ModelWrappers.ℓGradientResult","text":"struct ℓGradientResult{T, S, G} <: ℓObjectiveResult\n\nStores result for log density, gradient, and parameter for 'ℓobjective' evaluation at 'parameter'.\n\nFields\n\nθᵤ::Any\nParameter in unconstrained space.\nℓθᵤ::Any\nLog density at θᵤ.\n∇ℓθᵤ::Any\nGradient of log density at θᵤ.\n\n\n\n\n\n","category":"type"},{"location":"#ModelWrappers.ℓObjectiveResult","page":"Home","title":"ModelWrappers.ℓObjectiveResult","text":"abstract type ℓObjectiveResult <: BaytesCore.AbstractResult\n\nAbstract super type for AbstractDifferentiableObjective results.\n\n\n\n\n\n","category":"type"},{"location":"#Base.fill!-Tuple{ModelWrapper, NamedTuple}","page":"Home","title":"Base.fill!","text":"fill!(model, θ)\n\n\nInplace version of fill.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Base.fill-Tuple{ModelWrapper, NamedTuple}","page":"Home","title":"Base.fill","text":"fill(model, θ)\n\n\nFill 'model' values with NamedTuple 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Simplex_from_flatten!-Union{Tuple{R}, Tuple{AbstractVector{R}, Union{AbstractVector{R}, R}}} where R<:Real","page":"Home","title":"ModelWrappers.Simplex_from_flatten!","text":"Simplex_from_flatten!(buffer, x_vec)\n\n\nInplace version of Simplexfromflatten. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Simplex_from_flatten-Union{Tuple{Union{AbstractVector{R}, R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.Simplex_from_flatten","text":"Expand vector of k-1 dimensions back to k dimensions. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Symmetric_from_flatten!-Union{Tuple{R}, Tuple{T}, Tuple{AbstractMatrix{T}, Union{AbstractVector{R}, R}, BitMatrix}} where {T<:Real, R<:Real}","page":"Home","title":"ModelWrappers.Symmetric_from_flatten!","text":"Symmetric_from_flatten!(mat, x_vec, idx)\n\n\nInplace version of Symmetricfromflatten. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.Symmetric_from_flatten-Union{Tuple{R}, Tuple{Union{AbstractVector{R}, R}, BitMatrix}} where R<:Real","page":"Home","title":"ModelWrappers.Symmetric_from_flatten","text":"Symmetric_from_flatten(x_vec, idx)\n\n\nExpand vector back to (symmetric) Matrix. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._allparam-Tuple{Bool}","page":"Home","title":"ModelWrappers._allparam","text":"_allparam(val)\n\n\nReturns NamedTuple of true/false given parameter is not fixed. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._anyparam-Tuple{Bool}","page":"Home","title":"ModelWrappers._anyparam","text":"_anyparam(val)\n\n\nReturns NamedTuple of true/false given parameter is not fixed. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkfinite-Union{Tuple{R}, Tuple{T}, Tuple{T, R}} where {T<:Real, R<:Real}","page":"Home","title":"ModelWrappers._checkfinite","text":"_checkfinite(θ)\n_checkfinite(θ, max_val)\n\n\nCheck if 'θ' is of finite value and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkkeys-Union{Tuple{Ty}, Tuple{Ky}, Tuple{Tx}, Tuple{Kx}, Tuple{NamedTuple{Kx, Tx}, NamedTuple{Ky, Ty}}} where {Kx, Tx, Ky, Ty}","page":"Home","title":"ModelWrappers._checkkeys","text":"_checkkeys(x, y)\n\n\nCheck if all keys of 'x' and 'y' match - works with Nested Tuples - and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkparam-Tuple{Random.AbstractRNG, AbstractArray, AbstractArray}","page":"Home","title":"ModelWrappers._checkparam","text":"_checkparam(_rng, val, constraint)\n\n\nCheck if provided val-constraint combination is valid for Param struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkparams-Tuple{Any}","page":"Home","title":"ModelWrappers._checkparams","text":"_checkparams(param)\n\n\nCheck if all values in (Nested) NamedTuple are a 'Param' struct and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checkprior-Tuple{Any}","page":"Home","title":"ModelWrappers._checkprior","text":"_checkprior(prior)\n\n\nCheck if 'prior' is a valid density and return Bool. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._checksampleable-Tuple{Fixed}","page":"Home","title":"ModelWrappers._checksampleable","text":"_checksampleable(constraint)\n\n\nCheck if argument is not fixed. Returns NamedTuple with true/false. Needed in addition to _checkprior for nested NamedTuples. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._config-Union{Tuple{R}, Tuple{ModelWrappers.ADForward, Objective, AbstractVector{R}}} where R<:Real","page":"Home","title":"ModelWrappers._config","text":"_config(differentiation, objective, θᵤ)\n\n\nWrite config file for AD wrapper.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._diffresults_buffer-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:Real","page":"Home","title":"ModelWrappers._diffresults_buffer","text":"_diffresults_buffer(θᵤ)\n\n\nInitiate DiffResults.MutableDiffResult struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._get_constraint-Tuple{Param}","page":"Home","title":"ModelWrappers._get_constraint","text":"_get_constraint(param)\n\n\nRecursively collect constraints of 'Param' struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._get_val-Tuple{Param}","page":"Home","title":"ModelWrappers._get_val","text":"_get_val(param)\n\n\nRecursively collect values of 'Param' struct. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._paramnames-Tuple{Symbol, Integer}","page":"Home","title":"ModelWrappers._paramnames","text":"_paramnames(sym, len)\n\n\nReturn parameter names as a string in increasing order. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._to_bijector-Tuple{AbstractArray}","page":"Home","title":"ModelWrappers._to_bijector","text":"_to_bijector(infoᵥ)\n\n\nTransform user constraint to Bijector. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers._to_inv_bijector-Tuple{Bijectors.Bijector}","page":"Home","title":"ModelWrappers._to_inv_bijector","text":"_to_inv_bijector(info)\n\n\nTransform Bijector to its inverse. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.check_gradients","page":"Home","title":"ModelWrappers.check_gradients","text":"check_gradients(_rng, objective)\ncheck_gradients(_rng, objective, ADlibraries)\ncheck_gradients(_rng, objective, ADlibraries, θᵤ)\ncheck_gradients(_rng, objective, ADlibraries, θᵤ, difftune; printoutput)\n\n\nCheck gradient computations of different backends against 'objective'.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.constrain-Union{Tuple{T}, Tuple{S}, Tuple{S, T}} where {S<:Bijectors.Bijector, T}","page":"Home","title":"ModelWrappers.constrain","text":"Inverse-Transform 'θᵤ' into a constrained space given 'b⁻¹'. Returns same type as 'θᵤ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.dynamics-Tuple{Objective}","page":"Home","title":"ModelWrappers.dynamics","text":"dynamics(objective)\n\n\nAssign model dynamics for a given objective.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.fill_array!-Union{Tuple{F}, Tuple{T}, Tuple{AbstractArray{T}, Union{AbstractArray{F}, F}}} where {T<:Real, F<:Real}","page":"Home","title":"ModelWrappers.fill_array!","text":"fill_array!(buffer, vec)\n\n\nFill array with elements of vec. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten","page":"Home","title":"ModelWrappers.flatten","text":"flatten(x )\n\nConvert 'x' into a Vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.flatten-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.flatten","text":"flatten(model)\n\n\nFlatten 'model' values and return as vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten_Simplex-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_Simplex","text":"Flatten vector x to k-1 dimensions. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten_Symmetric-Union{Tuple{R}, Tuple{AbstractMatrix{R}, BitMatrix}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_Symmetric","text":"flatten_Symmetric(mat, idx)\n\n\nFlatten matrix to vector. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.flatten_array-Union{Tuple{AbstractArray{R}}, Tuple{R}} where R<:Real","page":"Home","title":"ModelWrappers.flatten_array","text":"flatten_array(mat)\n\n\nFlatten array x. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.generate-Tuple{Random.AbstractRNG, Objective}","page":"Home","title":"ModelWrappers.generate","text":"generate(_rng, objective)\n\n\nGenerate statistics given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_abs_det_jac-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_abs_det_jac","text":"log_abs_det_jac(model)\n\n\nEvaluate eventual Jacobian adjustments from transformations at 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_abs_det_jac-Union{Tuple{T}, Tuple{Bijectors.Identity, T}} where T","page":"Home","title":"ModelWrappers.log_abs_det_jac","text":"log_abs_det_jac(b, θ)\n\n\nEvaluate eventual Jacobian adjustments from transformation of 'b' at 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_density","page":"Home","title":"ModelWrappers.log_density","text":"log_density(objective, θᵤ)\n\nCompute log density of 'objective' at 'θᵤ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.log_density_and_gradient","page":"Home","title":"ModelWrappers.log_density_and_gradient","text":"log_density_and_gradient(objective, θᵤ)\n\nCompute log density and gradient of 'objective' at 'θᵤ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.log_density_and_hessian","page":"Home","title":"ModelWrappers.log_density_and_hessian","text":"log_density_and_hessian(objective, θᵤ)\n\nCompute log density, gradient and hessian of 'objective' at 'θᵤ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#ModelWrappers.log_prior-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_prior","text":"log_prior(model)\n\n\nEvaluate Log density of 'model' prior given current 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior-Union{Tuple{T}, Tuple{Any, T}} where T","page":"Home","title":"ModelWrappers.log_prior","text":"Evaluate Log density of 'prior' at 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior_with_transform-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.log_prior_with_transform","text":"log_prior_with_transform(model)\n\n\nEvaluate Log density and eventual Jacobian adjustments of 'model' prior given current 'model' values.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.log_prior_with_transform-Union{Tuple{T}, Tuple{Any, T}} where T","page":"Home","title":"ModelWrappers.log_prior_with_transform","text":"log_prior_with_transform(prior, θ)\n\n\nEvaluate Log density and eventual Jacobian adjustments from transformation of 'prior' at 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.paramcount-Union{Tuple{F}, Tuple{Symbol, F, Any}} where F<:FlattenDefault","page":"Home","title":"ModelWrappers.paramcount","text":"paramcount(sym, types, val)\n\n\nCount length of nested parameter tuple. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.paramnames-Union{Tuple{F}, Tuple{Symbol, F, Any, Any}} where F<:FlattenDefault","page":"Home","title":"ModelWrappers.paramnames","text":"paramnames(sym, types, val, constraint)\n\n\nReturn all parameter names in increasing order. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.predict-Tuple{Random.AbstractRNG, Objective}","page":"Home","title":"ModelWrappers.predict","text":"predict(_rng, objective)\n\n\nPredict new data given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.sample_constraint-Tuple{Random.AbstractRNG, Any}","page":"Home","title":"ModelWrappers.sample_constraint","text":"sample_constraint(_rng, prior)\n\n\nSample from constraint if 'prior'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.simulate-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.simulate","text":"simulate(model)\n\n\nSimulate data given Model parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.tag-Union{Tuple{AbstractMatrix{R}}, Tuple{R}, Tuple{AbstractMatrix{R}, Bool}, Tuple{AbstractMatrix{R}, Bool, Bool}} where R<:Real","page":"Home","title":"ModelWrappers.tag","text":"tag(mat)\ntag(mat, upper)\ntag(mat, upper, diag)\n\n\nAssign subset of elements to track in Matrix mat. Not exported.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.unconstrain","text":"unconstrain(model)\n\n\nUnconstrain 'model' values and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain-Union{Tuple{T}, Tuple{S}, Tuple{S, T}} where {S<:Bijectors.Bijector, T}","page":"Home","title":"ModelWrappers.unconstrain","text":"unconstrain(b, θ)\n\n\nTransform 'θ' into an unconstrained space given 'b'. Returns same type as 'θ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unconstrain_flatten-Tuple{ModelWrapper}","page":"Home","title":"ModelWrappers.unconstrain_flatten","text":"unconstrain_flatten(model)\n\n\nFlatten and unconstrain 'model' values and return as vector.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten!-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten!","text":"unflatten!(model, θ)\n\n\nInplace version of unflatten.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten","text":"Unlatten Vector 'θ' given constraints from 'model' and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten_constrain!-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten_constrain!","text":"Inplace version of unflatten_constrain.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ModelWrappers.unflatten_constrain-Union{Tuple{T}, Tuple{ModelWrapper, AbstractVector{T}}} where T<:Real","page":"Home","title":"ModelWrappers.unflatten_constrain","text":"Constrain and Unflatten vector 'θᵤ' given 'model' constraints.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample!-Tuple{Random.AbstractRNG, ModelWrapper}","page":"Home","title":"StatsBase.sample!","text":"sample!(_rng, model)\n\n\nInplace version of sample.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample-Tuple{Random.AbstractRNG, ModelWrapper}","page":"Home","title":"StatsBase.sample","text":"sample(_rng, model)\n\n\nSample from 'model' prior and return as NamedTuple.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
