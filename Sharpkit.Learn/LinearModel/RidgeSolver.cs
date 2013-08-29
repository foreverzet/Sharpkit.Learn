// -----------------------------------------------------------------------
// <copyright file="RidgeSolver.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;

    /// <summary>
    /// Solver to use in the Ridge computational routines.
    /// </summary>
    public enum RidgeSolver
    {
        /// <summary>
        /// Chooses the solver automatically based on the type of data.
        /// </summary>
        Auto,
        
        /// <summary>
        /// uses a Singular Value Decomposition of X to compute the Ridge
        ///  coefficients. More stable for singular matrices than <see cref="DenseCholesky"/>.
        /// </summary>
        Svd,

        /// <summary>
        /// Uses the standard Math.Net Cholesky() function to
        /// obtain a closed-form solution.
        /// </summary>
        DenseCholesky,

        /// <summary>
        /// Uses the dedicated regularized least-squares routine
        /// scipy.sparse.linalg.lsqr. It is the fatest but may not be available
        /// in old scipy versions. It also uses an iterative procedure.
        /// </summary>
        Lsqr
    }
}
