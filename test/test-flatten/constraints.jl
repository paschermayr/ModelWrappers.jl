############################################################################################
# Constraints

############################################################################################
@testset "Constraints - Bijector" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = 2.
            constraint = Bijectors.bijector(Gamma(2,2))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - Distribution" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = 2.
            constraint = Gamma(2,2)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - Constrained" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = 2.
            constraint = Constrained(1.,3.)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - Unconstrained" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = 2.
            constraint = Unconstrained()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - Fixed" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = zeros(2,3,4)
            constraint = Fixed()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
#            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - CorrelationMatrix" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = zeros(3,3)
            val[1,1] = val[2,2] = val[3,3] = 1.0
            val
            val[2,1] = val[1, 2] = 0.12
            val[3,1] = val[1, 3] = 0.13
            val[3,2] = val[2, 3] = 0.14
            val
            constraint = CorrelationMatrix()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat) == 3
            @test x_flat == output.([0.12, 0.13, 0.14])
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(4,4))
            @test_throws ArgumentError flatten(reconstruct, zeros(2,2))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(4,4))
            @test_throws ArgumentError flattenAD(reconstruct,  zeros(2,3))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            val_unconstrained = unconstrain(_transform, val)
            #Upper triangular
            @test val_unconstrained[3,1] == 0.0
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
        end
    end
end

############################################################################################
@testset "Constraints - CovarianceMatrix" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = zeros(3,3)
            val[1,1] = 1.5
            val[2,2] = 1.6
            val[3,3] = 1.7
            val
            val[2,1] = val[1, 2] = 0.12
            val[3,1] = val[1, 3] = 0.13
            val[3,2] = val[2, 3] = 0.14
            val
            constraint = CovarianceMatrix()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat) == 6
            @test x_flat == output.([1.50, 0.12, 0.13, 1.60, 0.14, 1.70])
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(4,4))
            @test_throws ArgumentError flatten(reconstruct, zeros(2,2))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(4,4))
            @test_throws ArgumentError flattenAD(reconstruct,  zeros(2,3))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            val_unconstrained = unconstrain(_transform, val)
            #Lower triangular
            @test val_unconstrained[1,3] == 0.0
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test val_constrained ≈ val atol = _TOL
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
################################################################################
# Custom Matrix flattening
            _tag = tag(val, false, true)
            _vals4 = flatten_Symmetric(val, _tag)
            @test length(_vals4) == 6
            @test all(Symmetric_from_flatten(_vals4, _tag) .≈ val)
        end
    end
end

############################################################################################
@testset "Constraints - Simplex" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [.1, .2, .7]
            constraint = Simplex(val)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat) == 2
            @test x_flat == output.([.1, .2])
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(4))
            @test_throws ArgumentError flatten(reconstruct, zeros(2))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(4))
            @test_throws ArgumentError flattenAD(reconstruct,  zeros(2))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Transforms
            _transform, _inversetransform = construct_transform(constraint, val)
            x_transformed = _transform(val)
            x_inversetransformed = _inversetransform(x_transformed)

            transformer = Transformconstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)

            @test transformer.unconstrain == _transform
            @test transformer.constrain == _inversetransform
            @test x_transformed == val_uncon
            @test x_inversetransformed == val_con

            @test x_inversetransformed isa typeof(val)
            val_unconstrained = unconstrain(_transform, val)
            val_constrained = constrain(_inversetransform, val_unconstrained)
            @test val_constrained isa typeof(val)
            @test sum(val_constrained .≈ val) == 3
            @test sum(val_constrained) ≈ 1.0
            logabs = log_abs_det_jac(_transform, val)
            @test logabs isa AbstractFloat
            @test _check(_RNG, constraint, val)
################################################################################
# Custom Matrix flattening
            _vals4 = flatten_Simplex(val)
            @test length(_vals4) == length(val)-1
            _vals5 = Simplex_from_flatten!(zeros(length(val)), _vals4)
            _vals6 = Simplex_from_flatten(_vals4)
            @test sum(_vals5) ≈ 1.0
            @test sum(_vals6) ≈ 1.0
            @test all(output.(val) .≈ _vals5)
            @test all(output.(val) .≈ _vals6)
        end
    end
end
