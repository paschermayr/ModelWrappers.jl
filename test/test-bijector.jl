############################################################################################
# Bijectors
############################################################################################

@testset "Bijector - Simplex" begin
    simplex = [.3, .4, .3]
    bij = Bijectors.SimplexBijector()
    inv_bij = inverse(bij)
# Flatten
    _vals, unflat = flatten(simplex, bij)
    @test length(_vals) == length(simplex)-1
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(simplex)
    @test all(_vals2 .≈ simplex)
# Constraining
    _valsᵤ = constrain(bij, simplex)
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test sum(_vals3) ≈ 1.0 atol = _TOL
# Custom Simplex flattening
    _vals4 = ModelWrappers.flatten_Simplex(simplex)
    @test length(_vals4) == length(simplex)-1
    _vals5 = ModelWrappers.Simplex_from_flatten!(zeros(length(simplex)), _vals4)
    _vals6 = ModelWrappers.Simplex_from_flatten(_vals4)
    @test all(simplex .≈ _vals5)
    @test all(simplex .≈ _vals6)
end

@testset "Bijector - PDMatrix" begin
    covmat = [1.2 .5 ; .5 3.4]
    bij = Bijectors.PDBijector()
    inv_bij = inverse(bij)
# Flatten
    _vals, unflat = flatten(covmat, bij)
    @test length(_vals) == 3
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(covmat)
    @test all(_vals2 .≈ covmat)
# Constraining
    _valsᵤ = constrain(bij, covmat)
    #Lower triangular - hopefully fixed...
    @test _valsᵤ[1, 2] == 0.0
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test all(_vals3 .≈ covmat)
# Custom Matrix flattening
    _tag = ModelWrappers.tag(covmat, false, true)
    _vals4 = ModelWrappers.flatten_Symmetric(covmat, _tag)
    @test length(_vals4) == 3
end

@testset "Bijector - CorrMatrix" begin
    cormat = [1. .5 ; .5 1.]
    bij = Bijectors.CorrBijector()
    inv_bij = inverse(bij)
# Flatten
    _vals, unflat = flatten(cormat, bij)
    @test length(_vals) == 1
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(cormat)
# Constraining
    _valsᵤ = constrain(bij, cormat)
    #Upper triangular
    @test _valsᵤ[1, 2] != 0.0
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test all(_vals3 .≈ cormat)
end
