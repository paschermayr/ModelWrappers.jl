############################################################################################
# Constraints

############################################################################################
@testset "Constraints - Bijecton" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = 2.
            constraint = Bijection(Bijectors.bijector(Gamma(2,2)))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test val_unflat == val
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test val_constrained == val
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test val_unflat_constrained ≈ val
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]


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
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
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

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(DistributionConstraint(constraint), val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test val_unflat == val
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test val_constrained == val
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test val_unflat_constrained ≈ val
        
            check_AD = check_AD_closure(DistributionConstraint(constraint), val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]

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
            transformer = TransformConstructor(DistributionConstraint(constraint), val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, DistributionConstraint(constraint), val)

            lp = logpdf(constraint, val)
            @test lp + logabs ≈ logpdf_with_trans(constraint, val, true)

################################################################################
# Check custom constraint function for distributions
            sample_constraint(_RNG, DistributionConstraint(constraint), val)
            @test log_prior(DistributionConstraint(constraint), val) ≈ log_prior(constraint, val)
#####################################
            constraint_custom = [DistributionConstraint(constraint), DistributionConstraint(constraint)]
            val_custom = [1., 5.]
            transformer = TransformConstructor(constraint_custom, val_custom)
            val_uncon = unconstrain(transformer, val_custom)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val_custom)
            @test val_con ≈ val_custom
            @test val_con isa typeof(val_custom)
            @test val_uncon isa typeof(val_custom)
            @test logabs isa AbstractFloat

            lp = logpdf(constraint_custom[1].dist, val_custom[1]) + logpdf(constraint_custom[2].dist, val_custom[2])
            lp_transform = logpdf_with_trans(constraint_custom[1].dist, val_custom[1], true) + logpdf_with_trans(constraint_custom[2].dist, val_custom[2], true)
            @test lp + logabs ≈ lp_transform
#####################################
            constraint_custom = (DistributionConstraint(constraint), DistributionConstraint(constraint), Fixed())
            val_custom = (1., 5., rand(1,2,3,4,5))
            transformer = TransformConstructor(constraint_custom, val_custom)
            val_uncon = unconstrain(transformer, val_custom)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val_custom)
            @test all(val_con .≈ val_custom)
            @test val_con isa typeof(val_custom)
            @test val_uncon isa typeof(val_custom)
            @test logabs isa AbstractFloat

            lp = logpdf(constraint_custom[1].dist, val_custom[1]) + logpdf(constraint_custom[2].dist, val_custom[2])
            lp_transform = logpdf_with_trans(constraint_custom[1].dist, val_custom[1], true) + logpdf_with_trans(constraint_custom[2].dist, val_custom[2], true)
            @test lp + logabs ≈ lp_transform

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

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test val_unflat == val
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test val_constrained == val
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test val_unflat_constrained == val
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]

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
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
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

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test val_unflat == val
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test val_constrained == val
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test val_unflat_constrained == val
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]

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
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
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

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test val_unflat == val
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test val_constrained == val
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test val_unflat_constrained == val
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]
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
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
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
            constraint = Bijection( Bijectors.bijector(LKJ(3,1)) )
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test sum(val_unflat .- val) ≈ 0 atol = _TOL
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test sum(val_constrained .- val) ≈ 0 atol = _TOL
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test sum(val_unflat_constrained .- val) ≈ 0 atol = _TOL
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]

# Flatten
            x_flat_all = flatten(reconstruct, val)
            x_constrained = unconstrain(constraint, val)

            x_flat = unconstrain_flatten(info, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat_all) == output
            @test length(x_flat) == 3
            @test x_flat ≈ x_constrained #output.([0.12, 0.13, 0.14])

            # Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat_all)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat_all)
            @test eltype(x_unflatAD) == eltype(x_flat_all)
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
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
  #          @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
            #Upper triangular
#            @test val_uncon[3,1] == 0.0

################################################################################
# Check if Cor distribution also defaults to constraint
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = zeros(3,3)
            val[1,1] = val[2,2] = val[3,3] = 1.0
            val
            val[2,1] = val[1, 2] = 0.12
            val[3,1] = val[1, 3] = 0.13
            val[3,2] = val[2, 3] = 0.14
            val
            con = Distributions.LKJ(3, 1.0)
            constraint = con
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
#            @test length(x_flat) == 3
#            @test x_flat == output.([0.12, 0.13, 0.14])

            constraint = DistributionConstraint(con)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
#           @test eltype(x_flat) == output
#           @test length(x_flat) == 3
#           @test x_flat == output.([0.12, 0.13, 0.14])
        end
    end
end

@testset "Constraints - Cholesky LKJ" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            _dist = LKJCholesky(3,1)
            constraint = Bijection( Bijectors.bijector(_dist) )
            val = rand(_RNG, _dist)

            @test ModelWrappers._checkfinite(val)

            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
#            @test sum(val_unflat.factors .- val.factors) ≈ 0 atol = _TOL
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
#            @test sum(val_constrained.factors .- val.factors) ≈ 0 atol = _TOL
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
#            @test sum(val_unflat_constrained.factors .- val.factors) ≈ 0 atol = _TOL
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]


# Flatten
            x_flat_all = flatten(reconstruct, val)
            x_constrained = unconstrain(constraint, val)
            x_flat = unconstrain_flatten(info, val)

            @test x_flat isa AbstractVector
            @test eltype(x_flat_all) == output
            @test length(x_flat) == 3
            @test x_flat ≈ x_constrained #output.([0.12, 0.13, 0.14])

            # Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat_all)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat_all)
            @test eltype(x_unflatAD) == eltype(x_flat_all)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

#            @test val_con ≈ val
            @test val_con isa typeof(val)
  #          @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)
            #Upper triangular
#            @test val_uncon[3,1] == 0.0

################################################################################
# Check if Cor distribution also defaults to constraint
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = zeros(3,3)
            val[1,1] = val[2,2] = val[3,3] = 1.0
            val
            val[2,1] = val[1, 2] = 0.12
            val[3,1] = val[1, 3] = 0.13
            val[3,2] = val[2, 3] = 0.14
            val
            con = Distributions.LKJ(3, 1.0)
            constraint = con
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
#            @test length(x_flat) == 3
#            @test x_flat == output.([0.12, 0.13, 0.14])

            constraint = DistributionConstraint(con)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
#           @test eltype(x_flat) == output
#           @test length(x_flat) == 3
#           @test x_flat == output.([0.12, 0.13, 0.14])
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
            constraint =  Bijection( Bijectors.bijector(InverseWishart(10, [2. .3 ; .3 3.])) )
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
   #         @test sum(val_unflat .- val) ≈ 0 atol = _TOL
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
 #           @test sum(val_constrained .- val) ≈ 0 atol = _TOL
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
    #        @test sum(val_unflat_constrained .- val) ≈ 0 atol = _TOL
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
#            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]


            # Flatten
            x_flat = flatten(reconstruct, val)
            x_constrained = unconstrain(constraint, val)
            x_flat_constrained = unconstrain_flatten(info, val)


            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat_constrained) == 6
#            @test x_flat == output.([1.50, 0.12, 0.13, 1.60, 0.14, 1.70])
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
# Transforms

            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

#            @test val_con ≈ val
            @test val_con isa typeof(val)
#            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)

            #Lower triangular
 #           @test val_uncon[1,3] == 0.0
################################################################################
# Custom Matrix flattening
            _tag = ModelWrappers.tag(val, false, true)
            _vals4 = ModelWrappers.flatten_Symmetric(val, _tag)
            @test length(_vals4) == 6
            @test all(ModelWrappers.Symmetric_from_flatten(_vals4, _tag) .≈ val)
################################################################################
# Check if Cov distribution also defaults to constraint
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
            con = Distributions.InverseWishart(10.0, [1.0 0.0 0.; 0.0 1.0 .0 ; 0. 0. .1])
            constraint = con
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
#            @test eltype(x_flat) == output
 #           @test length(x_flat) == 6
  #          @test x_flat == output.([1.50, 0.12, 0.13, 1.60, 0.14, 1.70])

            constraint = DistributionConstraint(con)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
#            @test x_flat isa AbstractVector
 #           @test eltype(x_flat) == output
  #          @test length(x_flat) == 6
   #         @test x_flat == output.([1.50, 0.12, 0.13, 1.60, 0.14, 1.70])

        end
    end
end

############################################################################################
@testset "Constraints - Simplex" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [.1, .2, .7]
            constraint = Bijection( Bijectors.bijector(Dirichlet(3,3)) )
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)

            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(FlattenDefault(), reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test sum(val_unflat .- val) ≈ 0 atol = _TOL
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
            @test sum(val_constrained .- val) ≈ 0 atol = _TOL
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test sum(val_unflat_constrained .- val) ≈ 0 atol = _TOL
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]
            
            # Flatten
            x_flat = flatten(reconstruct, val)
            x_constrained = unconstrain(constraint, val)
            x_flat_constrained = unconstrain_flatten(info, val)
            
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat) == 3
            @test length(x_flat_constrained) == 2

#            @test x_flat == output.([.1, .2])
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
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

#            @test val_con ≈ val
#            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test ModelWrappers._check(_RNG, constraint, val)

#            @test sum(val_con .≈ val) == 3
            @test sum(val_con) ≈ 1.0
################################################################################
# Custom Matrix flattening
            _vals4 = ModelWrappers.flatten_Simplex(val)
            @test length(_vals4) == length(val)-1
            _vals5 = ModelWrappers.Simplex_from_flatten!(zeros(length(val)), _vals4)
            _vals6 = ModelWrappers.Simplex_from_flatten(_vals4)
            @test sum(_vals5) ≈ 1.0
            @test sum(_vals6) ≈ 1.0
            @test all(output.(val) .≈ _vals5)
            @test all(output.(val) .≈ _vals6)

################################################################################
# Check if dirichlet distribution also defaults to Simplex
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [.1, .2, .7]
            constraint = DistributionConstraint(Dirichlet(val))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
#            @test eltype(x_flat) == output
#            @test length(x_flat) == 2
#           @test x_flat == output.([.1, .2])

            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [.1, .2, .7]
            constraint = Dirichlet(val)
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
 #           @test eltype(x_flat) == output
 #           @test length(x_flat) == 2
 #           @test x_flat == output.([.1, .2])
        end
    end
end

############################################################################################
############################################################################################
############################################################################################
# Check if MultiConstraint correctly dispatches for special methods that map into lower dimensional space
_μ = 1.5
_p = [.5, .4, .1]
_ρ = [1.0 0.25; 0.25 1.0]
_σ = [2.0 0.5; 0.5 3.0]

_gammma = Gamma(2,3)
_dirichlet = Dirichlet(3,3)
_iwish = Distributions.InverseWishart(10.0, [1.0 0.0 ; 0.0 1.0])
_lkj = Distributions.LKJ(2, 1.0)
#=
_constraint = (
    ## Standard Distribution
    _gammma, DistributionConstraint(_gammma),
    [_gammma, _gammma], [DistributionConstraint(_gammma), DistributionConstraint(_gammma)],
    [[_gammma, _gammma], [_gammma, _gammma]], [[DistributionConstraint(_gammma), DistributionConstraint(_gammma)], [DistributionConstraint(_gammma), DistributionConstraint(_gammma)]],
    ## Simplex
    _dirichlet, DistributionConstraint(_dirichlet),
    [_dirichlet, _dirichlet], [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)],
    [[_dirichlet, _dirichlet], [_dirichlet, _dirichlet]], [[DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)], [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)]],
    ## Correlation
    _lkj, DistributionConstraint(_lkj),
    [_lkj, _lkj], [DistributionConstraint(_lkj), DistributionConstraint(_lkj)],
    [[_lkj, _lkj], [_lkj, _lkj]], [[DistributionConstraint(_lkj), DistributionConstraint(_lkj)], [DistributionConstraint(_lkj), DistributionConstraint(_lkj)]],
    ## Covariance
    _iwish, DistributionConstraint(_iwish),
    [_iwish, _iwish], [DistributionConstraint(_iwish), DistributionConstraint(_iwish)],
    [[_iwish, _iwish], [_iwish, _iwish]], [[DistributionConstraint(_iwish), DistributionConstraint(_iwish)], [DistributionConstraint(_iwish), DistributionConstraint(_iwish)]],
);
=#
_constraint = (
    ## Standard Distribution
    DistributionConstraint(_gammma), DistributionConstraint(_gammma),
    [DistributionConstraint(_gammma), DistributionConstraint(_gammma)], [DistributionConstraint(_gammma), DistributionConstraint(_gammma)],
    [[DistributionConstraint(_gammma), DistributionConstraint(_gammma)], [DistributionConstraint(_gammma), DistributionConstraint(_gammma)]], [[DistributionConstraint(_gammma), DistributionConstraint(_gammma)], [DistributionConstraint(_gammma), DistributionConstraint(_gammma)]],
    ## Simplex
    DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet),
    [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)], [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)],
    [[DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)], [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)]], [[DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)], [DistributionConstraint(_dirichlet), DistributionConstraint(_dirichlet)]],
    ## Correlation
    DistributionConstraint(_lkj), DistributionConstraint(_lkj),
    [DistributionConstraint(_lkj), DistributionConstraint(_lkj)], [DistributionConstraint(_lkj), DistributionConstraint(_lkj)],
    [[DistributionConstraint(_lkj), DistributionConstraint(_lkj)], [DistributionConstraint(_lkj), DistributionConstraint(_lkj)]], [[DistributionConstraint(_lkj), DistributionConstraint(_lkj)], [DistributionConstraint(_lkj), DistributionConstraint(_lkj)]],
    ## Covariance
    DistributionConstraint(_iwish), DistributionConstraint(_iwish),
    [DistributionConstraint(_iwish), DistributionConstraint(_iwish)], [DistributionConstraint(_iwish), DistributionConstraint(_iwish)],
    [[DistributionConstraint(_iwish), DistributionConstraint(_iwish)], [DistributionConstraint(_iwish), DistributionConstraint(_iwish)]], [[DistributionConstraint(_iwish), DistributionConstraint(_iwish)], [DistributionConstraint(_iwish), DistributionConstraint(_iwish)]],
);

_val = (
    ## Standard Distribution
    copy(_μ), copy(_μ),
    [copy(_μ), copy(_μ)], [copy(_μ), copy(_μ)],
    [[copy(_μ), copy(_μ)], [copy(_μ), copy(_μ)]], [[copy(_μ), copy(_μ)], [copy(_μ), copy(_μ)]],
    ## Simplex
    copy(_p), copy(_p),
    [copy(_p), copy(_p)], [copy(_p), copy(_p)],
    [[copy(_p), copy(_p)], [copy(_p), copy(_p)]], [[copy(_p), copy(_p)], [copy(_p), copy(_p)]],
    ## Correlation
    copy(_ρ), copy(_ρ),
    [copy(_ρ), copy(_ρ)], [copy(_ρ), copy(_ρ)],
    [[copy(_ρ), copy(_ρ)], [copy(_ρ), copy(_ρ)]], [[copy(_ρ), copy(_ρ)], [copy(_ρ), copy(_ρ)]],
    ## Covariance
    copy(_σ), copy(_σ),
    [copy(_σ), copy(_σ)], [copy(_σ), copy(_σ)],
    [[copy(_σ), copy(_σ)], [copy(_σ), copy(_σ)]], [[copy(_σ), copy(_σ)], [copy(_σ), copy(_σ)]],
);

val_length_total = 14*1 + 14*3 + 14*4 + 14*4
val_length_reduced = 14*1 + 14*2 + 14*1 + 14*3

@testset "Constraints - MultiConstraint" begin
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = _val
            constraint = _constraint
            ReConstructor(constraint, val);
            reconstruct = ReConstructor(flatdefault, constraint, val);
            
            reconstruct = ReConstructor(flatdefault, constraint, val)
            transform = TransformConstructor(constraint, val)
            info = ParameterInfo(flatdefault, reconstruct, transform, val)
        
            val_flat = flatten(info, val)
            val_unflat = unflatten(info, val_flat)
            @test length(val_flat) == val_length_total
        
            val_unconstrained = unconstrain(info, val)
            val_constrained = constrain(info, val_unconstrained)
        
            val_flat_unconstrained = unconstrain_flatten(info, val)
            val_unflat_constrained = unflatten_constrain(info, val_flat_unconstrained)
            @test length(val_flat_unconstrained) == val_length_reduced
        
            check_AD = check_AD_closure(constraint, val)
            check_AD(val_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat_unconstrained)
            
#            grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, val_flat_unconstrained)[1]

# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
            @test length(x_flat) == val_length_total
            x_unflat = unflatten(reconstruct, x_flat)

# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test length(x_flatAD) == val_length_total
            @test x_flatAD isa AbstractVector
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)

# Constrain need information from Transformer, which is given in Param via ModelWrappers -> cannot test it without a Model, as Distribution constraint is transformed in step to create Model
            x_constrained = unconstrain(constraint, val)
            x_flat_constrained = unconstrain_flatten(info, val)

            @test length(x_flat) == val_length_total
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output

        end
    end
end

