using System;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.LeastSquares
{
    /// <summary>
    /// Sparse Equations and Least Squares.
    /// The original Fortran code was written by C. C. Paige and M. A. Saunders as
    /// described in
    /// C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
    /// equations and sparse least squares, TOMS 8(1), 43--71 (1982).
    /// C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
    /// equations and least-squares problems, TOMS 8(2), 195--209 (1982).
    /// It is licensed under the following BSD license:
    /// Copyright (c) 2006, Systems Optimization Laboratory
    /// All rights reserved.
    ///
    /// Redistribution and use in source and binary forms, with or without
    /// modification, are permitted provided that the following conditions are
    ///    met:
    ///
    ///    * Redistributions of source code must retain the above copyright
    ///  notice, this list of conditions and the following disclaimer.
    ///
    /// * Redistributions in binary form must reproduce the above
    ///   copyright notice, this list of conditions and the following
    ///   disclaimer in the documentation and/or other materials provided
    ///   with the distribution.
    /// * Neither the name of Stanford University nor the names of its
    ///   contributors may be used to endorse or promote products derived
    ///  from this software without specific prior written permission.
    /// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    /// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    /// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    /// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    /// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    /// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    /// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    /// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    /// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    /// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    /// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    /// The Fortran code was translated to Python for use in CVXOPT by Jeffery
    /// Kline with contributions by Mridul Aanjaneya and Bob Myhill.
    /// Adapted for SciPy by Stefan van der Walt.
    /// </summary>
    internal class Lsqr
    {
        /// <summary>
        /// Stable implementation of Givens rotation.
        ///     Notes
        ///    -----
        ///     The routine 'SymOrtho' was added for numerical stability. This is
        ///     recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
        ///     ``1/eps`` in some important places (see, for example text following
        /// "Compute the next plane rotation Qk" in minres.py).
        ///
        /// References
        /// ----------
        /// .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
        ///   and Least-Squares Problems", Dissertation,
        ///   http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
        /// </summary>
        /// <param name="?"></param>
        /// <param name="?"></param>
        private static Tuple<double, double, double> SymOrtho(double a, double b)
        {
            if (b == 0)
            {
                return Tuple.Create((double)Math.Sign(a), 0.0, Math.Abs(a));
            }
            
            if (a == 0)
            {
                return Tuple.Create(0.0, (double)Math.Sign(b), Math.Abs(b));
            }
            
            if (Math.Abs(b) > Math.Abs(a))
            {
                double tau = a/b;
                double s = Math.Sign(b)/Math.Sqrt(1 + tau*tau);
                double c = s*tau;
                double r = b/s;
                return Tuple.Create(c, s, r);
            }
            else
            {
                double tau = b/a;
                double c = Math.Sign(a)/Math.Sqrt(1 + tau*tau);
                double s = c*tau;
                double r = a/c;
                return Tuple.Create(c, s, r);
            }
        }

        /// <summary>
        /// Find the least-squares solution to a large, sparse, linear system
        /// of equations.
        ///
        /// The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
        /// ``min ||Ax - b||^2 + d^2 ||x||^2``.
        ///
        /// The matrix A may be square or rectangular (over-determined or
        /// under-determined), and may have any rank.
        ///
        /// ::
        /// 1. Unsymmetric equations --    solve  A*x = b
        ///
        /// 2. Linear least squares  --    solve  A*x = b
        ///                          in the least-squares sense
        ///
        /// 3. Damped least squares  --    solve  (   A    )*x = ( b )
        ///                                  ( damp*I )     ( 0 )
        ///                          in the least-squares sense
        /// </summary>
        /// <param name="a">
        /// Representation of an m-by-n matrix.  It is required that
        /// the linear operator can produce ``Ax`` and ``A^T x``.
        /// </param>
        /// <param name="b">Right-hand side vector ``b``.</param>
        /// <param name="damp">Damping coefficient.</param>
        /// <param name="atol">Stopping tolerance.</param>
        /// <param name="btol">Stopping tolerance. If both atol and btol are 1.0e-9 (say), the final
        /// residual norm should be accurate to about 9 digits.  (The
        /// final x will usually have fewer correct digits, depending on
        /// cond(A) and the size of damp.)
        /// </param>
        /// <param name="conlim">
        /// Another stopping tolerance.  lsqr terminates if an estimate of
        /// ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
        /// b``, `conlim` could be as large as 1.0e+12 (say).  For
        /// least-squares problems, conlim should be less than 1.0e+8.
        /// Maximum precision can be obtained by setting ``atol = btol =
        /// conlim = zero``, but the number of iterations may then be
        /// excessive.
        /// </param>
        /// <param name="iterLim">Explicit limitation on number of iterations (for safety).</param>
        /// <param name="show">Display an iteration log.</param>
        /// <param name="calcVar">Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.</param>
        /// <returns>Instance of <see cref="LsqrResult"/>.</returns>
        /// <remarks>
        /// LSQR uses an iterative method to approximate the solution.  The
        /// number of iterations required to reach a certain accuracy depends
        /// strongly on the scaling of the problem.  Poor scaling of the rows
        /// or columns of A should therefore be avoided where possible.
        /// 
        /// For example, in problem 1 the solution is unaltered by
        /// row-scaling.  If a row of A is very small or large compared to
        /// the other rows of A, the corresponding row of ( A  b ) should be
        /// scaled up or down.
        ///
        /// In problems 1 and 2, the solution x is easily recovered
        /// following column-scaling.  Unless better information is known,
        /// the nonzero columns of A should be scaled so that they all have
        /// the same Euclidean norm (e.g., 1.0).
        ///
        /// In problem 3, there is no freedom to re-scale if damp is
        /// nonzero.  However, the value of damp should be assigned only
        /// after attention has been paid to the scaling of A.
        ///
        /// The parameter damp is intended to help regularize
        /// ill-conditioned systems, by preventing the true solution from
        /// being very large.  Another aid to regularization is provided by
        /// the parameter acond, which may be used to terminate iterations
        /// before the computed solution becomes very large.
        ///
        /// If some initial estimate ``x0`` is known and if ``damp == 0``,
        /// one could proceed as follows:
        ///
        /// 1. Compute a residual vector ``r0 = b - A*x0``.
        /// 2. Use LSQR to solve the system  ``A*dx = r0``.
        /// 3. Add the correction dx to obtain a final solution ``x = x0 + dx``.
        ///
        /// This requires that ``x0`` be available before and after the call
        /// to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
        /// to solve A*x = b and k2 iterations to solve A*dx = r0.
        /// If x0 is "good", norm(r0) will be smaller than norm(b).
        /// If the same stopping tolerances atol and btol are used for each
        /// system, k1 and k2 will be similar, but the final solution x0 + dx
        /// should be more accurate.  The only way to reduce the total work
        /// is to use a larger stopping tolerance for the second system.
        /// If some value btol is suitable for A*x = b, the larger value
        /// btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
        ///
        /// Preconditioning is another way to reduce the number of iterations.
        /// If it is possible to solve a related system ``M*x = b``
        /// efficiently, where M approximates A in some helpful way (e.g. M -
        /// A has low rank or its elements are small relative to those of A),
        /// LSQR may converge more rapidly on the system ``A*M(inverse)*z =
        /// b``, after which x can be recovered by solving M*x = z.
        ///
        /// If A is symmetric, LSQR should not be used!
        ///
        /// Alternatives are the symmetric conjugate-gradient method (cg)
        /// and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
        /// applies to any symmetric A and will converge more rapidly than
        /// LSQR.  If A is positive definite, there are other implementations
        /// of symmetric cg that require slightly less work per iteration than
        /// SYMMLQ (but will take the same number of iterations).
        ///
        /// References
        /// ----------
        /// .. [1] C. C. Paige and M. A. Saunders (1982a).
        ///   "LSQR: An algorithm for sparse linear equations and
        ///   sparse least squares", ACM TOMS 8(1), 43-71.
        /// .. [2] C. C. Paige and M. A. Saunders (1982b).
        ///   "Algorithm 583.  LSQR: Sparse linear equations and least
        ///   squares problems", ACM TOMS 8(2), 195-209.
        ///.. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
        ///   systems using LSQR and CRAIG", BIT 35, 588-604.
        /// </remarks>
        public static LsqrResult lsqr(
            Matrix a,
            Vector b,
            double damp = 0.0,
            double atol = 1E-8,
            double btol = 1E-8,
            double conlim = 1e8,
            int? iterLim = null,
            bool show = false,
            bool calcVar = false)
        {
            int m = a.RowCount;
            int n = a.ColumnCount;
            if (iterLim == null)
            {
                iterLim = 2*n;
            }

            Vector @var = DenseVector.Create(n, (i) => 0.0);

            string[] msg = new[]
                               {
                                   "The exact solution is  x = 0                              ",
                                   "Ax - b is small enough, given atol, btol                  ",
                                   "The least-squares solution is good enough, given atol     ",
                                   "The estimate of cond(Abar) has exceeded conlim            ",
                                   "Ax - b is small enough for this machine                   ",
                                   "The least-squares solution is good enough for this machine",
                                   "Cond(Abar) seems to be too large for this machine         ",
                                   "The iteration limit has been reached                      "
                               };

            if (show)
            {
                Console.WriteLine(" ");
                Console.WriteLine("LSQR            Least-squares solution of  Ax = b");
                Console.WriteLine("The matrix A has {0} rows  and {1} cols'", m, n);
                Console.WriteLine("damp = {0}   calc_var = {1}", damp, calcVar);
                Console.WriteLine("atol = {0}   conlim = {1}", atol, conlim);
                Console.WriteLine("btol = {0}   iter_lim = {1}", btol, iterLim);
            }

            int itn = 0;
            int istop = 0;
            int nstop = 0;
            double ctol = 0;
            if (conlim > 0)
            {
                ctol = 1/conlim;
            }

            double anorm = 0;
            double acond = 0;
            double dampsq = damp*damp;
            double ddnorm = 0;
            double res2 = 0;
            double xnorm = 0;
            double xxnorm = 0;
            double z = 0;
            double cs2 = -1;
            double sn2 = 0;


            // Set up the first vectors u and v for the bidiagonalization.
            // These satisfy  beta*u = b,  alfa*v = A'u.
            Vector v = DenseVector.Create(n, i => 0.0);
            Vector u = b;
            Vector x = DenseVector.Create(n, i => 0.0);
            double alfa = 0;
            double beta = u.FrobeniusNorm();
            Vector w = DenseVector.Create(n, i => 0.0);

            if (beta > 0)
            {
                u.Multiply(1/beta, u);
                v = (Vector)(u*a);
                alfa = v.FrobeniusNorm();
            }

            if (alfa > 0)
            {
                v = (Vector)(v*(1/alfa));
                w = (Vector)v.Clone();
            }

            double rhobar = alfa;
            double phibar = beta;
            double bnorm = beta;
            double rnorm = beta;
            double r1norm = rnorm;
            double r2norm = rnorm;

            // Reverse the order here from the original matlab code because
            // there was an error on return when arnorm==0
            double arnorm = alfa*beta;
            if (arnorm == 0)
            {
                Console.WriteLine(msg[0]);
                return new LsqrResult
                           {
                               X = x,
                               IsStop = istop,
                               ItN = itn,
                               R1Norm = r1norm,
                               R2Norm = r2norm,
                               ANorm = anorm,
                               ACond = acond,
                               ArNorm = arnorm,
                               XNorm = xnorm,
                               Var = var
                           };
            }

            string head1 = "   Itn      x[0]       r1norm     r2norm ";
            string head2 = " Compatible    LS      Norm A   Cond A";

            if (show)
            {
                Console.WriteLine(" ");
                Console.WriteLine(head1 + head2);
                double test1 = 1;
                double test2 = alfa/beta;
                Console.WriteLine("{0} {1} {2} {3} {4} {5}", itn, x[0], r1norm, r2norm, test1, test2);
            }

            // Main iteration loop.
            while (itn < iterLim)
            {
                itn = itn + 1;

                // Perform the next step of the bidiagonalization to obtain the
                // next  beta, u, alfa, v.  These satisfy the relations
                // beta*u  =  a*v   -  alfa*u,
                // alfa*v  =  A'*u  -  beta*v.
                u = (Vector)(a*v - alfa*u);
                beta = u.ToDenseMatrix().FrobeniusNorm();

                if (beta > 0)
                {
                    u = (Vector)((1/beta)*u);
                    anorm = Math.Sqrt(anorm*anorm + alfa*alfa + beta*beta + damp*damp);
                    v = (Vector)((u*a) - beta*v);
                    alfa = v.ToDenseMatrix().FrobeniusNorm();
                    if (alfa > 0)
                    {
                        v.Multiply(1/alfa, v);
                    }
                }

                // Use a plane rotation to eliminate the damping parameter.
                // This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
                double rhobar1 = Math.Sqrt(rhobar*rhobar + damp*damp);
                double cs1 = rhobar/rhobar1;
                double sn1 = damp/rhobar1;
                double psi = sn1*phibar;
                phibar = cs1*phibar;

                // Use a plane rotation to eliminate the subdiagonal element (beta)
                // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
                var res = SymOrtho(rhobar1, beta);
                double cs = res.Item1;
                double sn = res.Item2;
                double rho = res.Item3;

                double theta = sn*alfa;
                rhobar = -cs*alfa;
                double phi = cs*phibar;
                phibar = sn*phibar;
                double tau = sn*phi;

                // Update x and w.
                double t1 = phi/rho;
                double t2 = -theta/rho;
                Vector dk = (Vector)w.Multiply(1/rho);

                x.Add(w.Multiply(t1), x);
                w = (Vector)(v + w*t2);
                ddnorm = ddnorm + Math.Pow(dk.FrobeniusNorm(), 2);

                if (calcVar)
                {
                    @var = (Vector)(@var + dk*dk);
                }

                // Use a plane rotation on the right to eliminate the
                // super-diagonal element (theta) of the upper-bidiagonal matrix.
                // Then use the result to estimate norm(x).
                double delta = sn2*rho;
                double gambar = -cs2*rho;
                double rhs = phi - delta*z;
                double zbar = rhs/gambar;
                xnorm = Math.Sqrt(xxnorm + zbar*zbar);
                double gamma = Math.Sqrt(gambar*gambar + theta*theta);
                cs2 = gambar/gamma;
                sn2 = theta/gamma;
                z = rhs/gamma;
                xxnorm = xxnorm + z*z;

                // Test for convergence.
                // First, estimate the condition of the matrix  Abar,
                // and the norms of  rbar  and  Abar'rbar.
                acond = anorm*Math.Sqrt(ddnorm);
                double res1 = phibar*phibar;
                res2 = res2 + psi*psi;
                rnorm = Math.Sqrt(res1 + res2);
                arnorm = alfa*Math.Abs(tau);

                // Distinguish between
                //    r1norm = ||b - Ax|| and
                //    r2norm = rnorm in current code
                //           = sqrt(r1norm^2 + damp^2*||x||^2).
                //    Estimate r1norm from
                //    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
                // Although there is cancellation, it might be accurate enough.
                double r1sq = rnorm*rnorm - dampsq*xxnorm;
                r1norm = Math.Sqrt(Math.Abs(r1sq));
                if (r1sq < 0)
                {
                    r1norm = -r1norm;
                }

                r2norm = rnorm;


                // Now use these norms to estimate certain other quantities,
                // some of which will be small near a solution.
                double test1 = rnorm/bnorm;
                double test2 = arnorm/(anorm*rnorm);
                double test3 = 1/acond;
                t1 = test1/(1 + anorm*xnorm/bnorm);
                double rtol = btol + atol*anorm*xnorm/bnorm;

                // The following tests guard against extremely small values of
                // atol, btol  or  ctol.  (The user may have set any or all of
                // the parameters  atol, btol, conlim  to 0.)
                // The effect is equivalent to the normal tests using
                // atol = eps,  btol = eps,  conlim = 1/eps.
                if (itn >= iterLim)
                    istop = 7;
                if (1 + test3 <= 1)
                    istop = 6;
                if (1 + test2 <= 1)
                    istop = 5;
                if (1 + t1 <= 1)
                    istop = 4;

                // Allow for tolerances set by the user.
                if (test3 <= ctol)
                    istop = 3;
                if (test2 <= atol)
                    istop = 2;
                if (test1 <= rtol)
                    istop = 1;

                // See if it is time to print something.
                bool prnt = false;
                if (n <= 40)
                    prnt = true;
                if (itn <= 10)
                    prnt = true;
                if (itn >= iterLim - 10)
                    prnt = true;
                // if itn%10 == 0: prnt = True
                if (test3 <= 2*ctol)
                    prnt = true;
                if (test2 <= 10*atol)
                    prnt = true;
                if (test1 <= 10*rtol)
                    prnt = true;
                if (istop != 0)
                    prnt = true;

                if (prnt && show)
                {
                    Console.WriteLine(
                        "{0} {1} {2} {3} {4} {5} {6} {7}",
                        itn,
                        x[0],
                        r1norm,
                        r2norm,
                        test1,
                        test2,
                        anorm,
                        acond);
                }

                if (istop != 0)
                {
                    break;
                }

                // End of iteration loop.
                // Print the stopping condition.
                if (show)
                {
                    Console.WriteLine();
                    Console.WriteLine("LSQR finished");
                    Console.WriteLine(msg[istop]);
                    Console.WriteLine();
                    Console.WriteLine("istop ={0}   r1norm ={1} anorm ={2}   arnorm ={3}", istop, r1norm, anorm, arnorm);
                    Console.WriteLine("itn = {0} r2norm = {1} acond = {2} xnorm={3}", itn, r2norm, acond, xnorm);
                    Console.WriteLine();
                }
            }

            return new LsqrResult
                       {
                           X = x,
                           IsStop = istop,
                           ItN = itn,
                           R1Norm = r1norm,
                           R2Norm = r2norm,
                           ANorm = anorm,
                           ACond = acond,
                           ArNorm = arnorm,
                           XNorm = xnorm,
                           Var = var
                       };
        }
    }
}
