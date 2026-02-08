# Load common test setup (handles both installed package and standalone modes)
include(joinpath(@__DIR__, "common.jl"))

@testset "Parallel Scan" begin
    @testset "Matrix Affine Transform Composition" begin
        # Test that composition satisfies: (J₂, u₂) ⊕ (J₁, u₁) = (J₂*J₁, J₂*u₁ + u₂)
        D = 3
        J1 = rand(D, D)
        u1 = rand(D)
        J2 = rand(D, D)
        u2 = rand(D)

        t1 = MatrixAffineTransform(J1, u1)
        t2 = MatrixAffineTransform(J2, u2)

        t_composed = compose(t2, t1)

        @test t_composed.J ≈ J2 * J1
        @test t_composed.u ≈ J2 * u1 + u2
    end

    @testset "Matrix Affine Transform Application" begin
        D = 3
        J = rand(D, D)
        u = rand(D)
        x = rand(D)

        t = MatrixAffineTransform(J, u)
        result = apply(t, x)

        @test result ≈ J * x + u
    end

    @testset "Diagonal Affine Transform Composition" begin
        D = 5
        d1 = rand(D)
        u1 = rand(D)
        d2 = rand(D)
        u2 = rand(D)

        t1 = DiagonalAffineTransform(d1, u1)
        t2 = DiagonalAffineTransform(d2, u2)

        t_composed = compose(t2, t1)

        @test t_composed.d ≈ d2 .* d1
        @test t_composed.u ≈ d2 .* u1 + u2
    end

    @testset "Diagonal Affine Transform Application" begin
        D = 5
        d = rand(D)
        u = rand(D)
        x = rand(D)

        t = DiagonalAffineTransform(d, u)
        result = apply(t, x)

        @test result ≈ d .* x + u
    end

    @testset "Block 2x2 Affine Transform Composition" begin
        # Test composition of 2×2 blocks per dimension
        D = 4

        # Create two transforms
        a1, b1, c1, e1 = rand(D), rand(D), rand(D), rand(D)
        u1_x, u1_v = rand(D), rand(D)
        t1 = Block2x2AffineTransform(a1, b1, c1, e1, u1_x, u1_v)

        a2, b2, c2, e2 = rand(D), rand(D), rand(D), rand(D)
        u2_x, u2_v = rand(D), rand(D)
        t2 = Block2x2AffineTransform(a2, b2, c2, e2, u2_x, u2_v)

        t_composed = compose(t2, t1)

        # Verify by computing full block matrix product for one dimension
        for i in 1:D
            M1 = [a1[i] b1[i]; c1[i] e1[i]]
            M2 = [a2[i] b2[i]; c2[i] e2[i]]
            M_prod = M2 * M1

            @test t_composed.a[i] ≈ M_prod[1, 1]
            @test t_composed.b[i] ≈ M_prod[1, 2]
            @test t_composed.c[i] ≈ M_prod[2, 1]
            @test t_composed.e[i] ≈ M_prod[2, 2]

            # Verify offset
            u1_vec = [u1_x[i], u1_v[i]]
            u2_vec = [u2_x[i], u2_v[i]]
            u_composed = M2 * u1_vec + u2_vec

            @test t_composed.u_x[i] ≈ u_composed[1]
            @test t_composed.u_v[i] ≈ u_composed[2]
        end
    end

    @testset "Block 2x2 Affine Transform Application" begin
        D = 4
        a, b, c, e = ones(D), 0.1 .* ones(D), 0.2 .* rand(D), ones(D) .+ 0.01 .* rand(D)
        u_x, u_v = rand(D), rand(D)

        t = Block2x2AffineTransform(a, b, c, e, u_x, u_v)

        x_pos = rand(D)
        x_mom = rand(D)
        x = vcat(x_pos, x_mom)

        result = apply(t, x)

        # Verify per dimension
        for i in 1:D
            expected_pos = a[i] * x_pos[i] + b[i] * x_mom[i] + u_x[i]
            expected_mom = c[i] * x_pos[i] + e[i] * x_mom[i] + u_v[i]

            @test result[i] ≈ expected_pos
            @test result[D + i] ≈ expected_mom
        end
    end

    @testset "Leapfrog Transform Construction" begin
        # Test that Block2x2AffineTransform(H_diag, ε) creates correct leapfrog Jacobian
        D = 3
        H_diag = rand(D) .- 0.5  # Hessian diagonal (can be negative)
        ε = 0.1

        t = Block2x2AffineTransform(H_diag, ε)

        @test all(t.a .≈ 1.0)
        @test all(t.b .≈ ε)
        @test t.c ≈ ε .* H_diag
        @test t.e ≈ ones(D) .+ (ε^2) .* H_diag
        @test all(t.u_x .≈ 0.0)
        @test all(t.u_v .≈ 0.0)
    end

    @testset "Parallel Scan vs Sequential Scan - Matrix" begin
        # Verify parallel scan matches sequential scan for full matrices
        T_len = 20
        D = 3
        s0 = rand(D)

        # Create random transforms
        transforms = [MatrixAffineTransform(rand(D, D), rand(D)) for _ in 1:T_len]

        # Compute both ways
        traj_parallel = parallel_scan(transforms, s0)
        traj_sequential = sequential_scan(transforms, s0)

        @test size(traj_parallel) == (T_len, D)
        @test traj_parallel ≈ traj_sequential
    end

    @testset "Parallel Scan vs Sequential Scan - Diagonal" begin
        # Verify parallel scan matches sequential scan for diagonal case
        T_len = 50
        D = 10
        s0 = rand(D)

        # Create random diagonal transforms
        transforms = [DiagonalAffineTransform(rand(D), rand(D)) for _ in 1:T_len]

        # Compute both ways
        traj_parallel = parallel_scan(transforms, s0)
        traj_sequential = sequential_scan(transforms, s0)

        @test size(traj_parallel) == (T_len, D)
        @test traj_parallel ≈ traj_sequential
    end

    @testset "Parallel Scan vs Sequential Scan - Block 2x2" begin
        # Verify parallel scan matches sequential scan for block case
        T_len = 30
        D = 5  # position dimension (total state dimension is 2D)
        s0 = rand(2D)

        # Create random block transforms
        transforms = [
            Block2x2AffineTransform(rand(D), rand(D), rand(D), rand(D), rand(D), rand(D))
            for _ in 1:T_len
        ]

        # Compute both ways
        traj_parallel = parallel_scan(transforms, s0)
        traj_sequential = sequential_scan(transforms, s0)

        @test size(traj_parallel) == (T_len, 2D)
        @test traj_parallel ≈ traj_sequential
    end

    @testset "Convenience Functions - make_matrix_transforms" begin
        T_len = 10
        D = 3

        J = rand(T_len, D, D)
        u = rand(T_len, D)

        transforms = make_matrix_transforms(J, u)

        @test length(transforms) == T_len
        @test all(transforms[t].J ≈ J[t, :, :] for t in 1:T_len)
        @test all(transforms[t].u ≈ u[t, :] for t in 1:T_len)
    end

    @testset "Convenience Functions - make_diagonal_transforms" begin
        T_len = 10
        D = 5

        d = rand(T_len, D)
        u = rand(T_len, D)

        transforms = make_diagonal_transforms(d, u)

        @test length(transforms) == T_len
        @test all(transforms[t].d ≈ d[t, :] for t in 1:T_len)
        @test all(transforms[t].u ≈ u[t, :] for t in 1:T_len)
    end

    @testset "Convenience Functions - make_block_transforms" begin
        T_len = 10
        D = 4
        ε = 0.1

        H_diag = rand(T_len, D)
        u_x = rand(T_len, D)
        u_v = rand(T_len, D)

        transforms = make_block_transforms(H_diag, ε, u_x, u_v)

        @test length(transforms) == T_len
        for t in 1:T_len
            @test all(transforms[t].a .≈ 1.0)
            @test all(transforms[t].b .≈ ε)
            @test transforms[t].c ≈ ε .* H_diag[t, :]
            @test transforms[t].e ≈ ones(D) .+ (ε^2) .* H_diag[t, :]
            @test transforms[t].u_x ≈ u_x[t, :]
            @test transforms[t].u_v ≈ u_v[t, :]
        end
    end

    @testset "In-place Parallel Scan" begin
        T_len = 20
        D = 5
        s0 = rand(D)

        transforms = [DiagonalAffineTransform(rand(D), rand(D)) for _ in 1:T_len]

        # Allocate output
        trajectory = zeros(T_len, D)

        # Run in-place version
        parallel_scan!(trajectory, transforms, s0)

        # Compare with regular version
        traj_expected = parallel_scan(transforms, s0)

        @test trajectory ≈ traj_expected
    end

    @testset "Identity Transform Composition" begin
        D = 4

        # Matrix identity
        t_mat = MatrixAffineTransform(rand(D, D), rand(D))
        id_mat = IdentityMatrixTransform{Float64,Int}(D)

        @test compose(id_mat, t_mat) === t_mat
        @test compose(t_mat, id_mat) === t_mat
        @test compose(id_mat, id_mat) === id_mat

        # Diagonal identity
        t_diag = DiagonalAffineTransform(rand(D), rand(D))
        id_diag = IdentityDiagonalTransform{Float64,Int}(D)

        @test compose(id_diag, t_diag) === t_diag
        @test compose(t_diag, id_diag) === t_diag
        @test compose(id_diag, id_diag) === id_diag

        # Block identity
        t_block = Block2x2AffineTransform(rand(D), rand(D), rand(D), rand(D), rand(D), rand(D))
        id_block = IdentityBlockTransform{Float64,Int}(D)

        @test compose(id_block, t_block) === t_block
        @test compose(t_block, id_block) === t_block
        @test compose(id_block, id_block) === id_block
    end

    @testset "Identity Transform Application" begin
        D = 4
        x = rand(D)

        @test apply(IdentityMatrixTransform{Float64,Int}(D), x) === x
        @test apply(IdentityDiagonalTransform{Float64,Int}(D), x) === x

        x_block = rand(2D)
        @test apply(IdentityBlockTransform{Float64,Int}(D), x_block) === x_block
    end

    @testset "Numerical Stability - Long Chain" begin
        # Test that parallel scan remains numerically stable for longer chains
        T_len = 1000
        D = 5
        s0 = rand(D)

        # Use stable diagonal values (< 1 in magnitude to prevent explosion)
        transforms = [
            DiagonalAffineTransform(0.9 .* rand(D) .+ 0.05, rand(D))  # d ∈ [0.05, 0.95]
            for _ in 1:T_len
        ]

        traj_parallel = parallel_scan(transforms, s0)
        traj_sequential = sequential_scan(transforms, s0)

        # Should still match (within reasonable tolerance for accumulation)
        @test isapprox(traj_parallel, traj_sequential, rtol=1e-10)
    end

    @testset "Associativity of Composition" begin
        # Verify (t3 ∘ t2) ∘ t1 = t3 ∘ (t2 ∘ t1)
        D = 3

        # Matrix case
        t1 = MatrixAffineTransform(rand(D, D), rand(D))
        t2 = MatrixAffineTransform(rand(D, D), rand(D))
        t3 = MatrixAffineTransform(rand(D, D), rand(D))

        left = compose(compose(t3, t2), t1)
        right = compose(t3, compose(t2, t1))

        @test left.J ≈ right.J
        @test left.u ≈ right.u

        # Diagonal case
        d1 = DiagonalAffineTransform(rand(D), rand(D))
        d2 = DiagonalAffineTransform(rand(D), rand(D))
        d3 = DiagonalAffineTransform(rand(D), rand(D))

        left_d = compose(compose(d3, d2), d1)
        right_d = compose(d3, compose(d2, d1))

        @test left_d.d ≈ right_d.d
        @test left_d.u ≈ right_d.u

        # Block case
        b1 = Block2x2AffineTransform(rand(D), rand(D), rand(D), rand(D), rand(D), rand(D))
        b2 = Block2x2AffineTransform(rand(D), rand(D), rand(D), rand(D), rand(D), rand(D))
        b3 = Block2x2AffineTransform(rand(D), rand(D), rand(D), rand(D), rand(D), rand(D))

        left_b = compose(compose(b3, b2), b1)
        right_b = compose(b3, compose(b2, b1))

        @test left_b.a ≈ right_b.a
        @test left_b.b ≈ right_b.b
        @test left_b.c ≈ right_b.c
        @test left_b.e ≈ right_b.e
        @test left_b.u_x ≈ right_b.u_x
        @test left_b.u_v ≈ right_b.u_v
    end
end
