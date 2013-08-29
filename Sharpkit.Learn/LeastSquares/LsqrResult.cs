using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.LeastSquares
{
    internal class LsqrResult
    {
        /// <summary>
        /// The final solution.
        /// </summary>
        public Vector X { get; set; }
            
        /// <summary>
        /// Gives the reason for termination.
        /// 1 means x is an approximate solution to Ax = b.
        /// 2 means x approximately solves the least-squares problem.
        /// </summary>
        public int IsStop { get; set; }

        /// <summary>
        /// Iteration number upon termination.
        /// </summary>
        public int ItN { get; set; }

        /// <summary>
        /// ``norm(r)``, where ``r = b - Ax``.
        /// </summary>
        public double R1Norm { get; set; }

        /// <summary>
        /// ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
        /// ``damp == 0``.
        /// </summary>
        public double R2Norm { get; set; }

        /// <summary>
        /// Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
        /// </summary>
        public double ANorm { get; set; }

        /// <summary>
        /// Estimate of ``cond(Abar)``.
        /// </summary>
        public double ACond { get; set; }

        /// <summary>
        /// Estimate of ``norm(A'*r - damp^2*x)``.
        /// </summary>
        public double ArNorm { get; set; }

        /// <summary>
        /// ``norm(x)``
        /// </summary>
        public double XNorm { get; set; }

        /// <summary>
        /// If ``calcVar`` is True, estimates all diagonals of
        /// ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
        /// damp^2*I)^{-1}``.  This is well defined if A has full column
        /// rank or ``damp > 0``.  (Not sure what var means if ``rank(A) < n`` and ``damp = 0.``)
        /// </summary>
        public Vector Var { get; set; }
    }
}