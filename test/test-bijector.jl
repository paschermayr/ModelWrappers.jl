#Make sure that Bijector formulations are in sync with ModelWrappers

@testset "Bijectors - Ordered implementation" begin
    val = [1., 2., 3.]
    d = Bijectors.ordered(MvNormal(I * [1., 1.1, 1.2], LinearAlgebra.Diagonal(ones(3))))
    b = Bijectors.bijector(d)
    b_inv = inverse(b)

    val_u = b(val)
    val_orig = b_inv(val_u)
    @test all( val_orig .== val)

    theta_u = [10., -10., 5.]
    theta_orig = b_inv(theta_u)
    @test theta_orig[1] < theta_orig[2] < theta_orig[3]
end


@testset "Bijectors - log_abs_det_jac formulation" begin
    val = 10.0
    d = Gamma(1,2)

    b = Bijectors.bijector(d)
    b_inv = inverse(b)
    valᵤ = b(val)

    ℓd = logpdf(d, val)
    logabsdet = logabsdetjac(b, val)
    ℓdᵤ = logpdf_with_trans(d, val, true)
    @test ℓd + (-logabsdet) == ℓdᵤ
end

@testset "Bijectors - transform function" begin
    val = [.2, .3, .5]
    d = Dirichlet(3,3)

    mybi = Bijectors.bijector(d)
    myinvbi = inverse(mybi)
    valᵤ = Bijectors.transform(mybi, val)
    @test all( mybi(val) .== Bijectors.transform(mybi, val) )
    @test all( myinvbi(valᵤ) .≈ Bijectors.transform(myinvbi, valᵤ) )
end
