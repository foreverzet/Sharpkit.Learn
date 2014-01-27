// -----------------------------------------------------------------------
// <copyright file="Splitter.cs" company="Sharpkit.Learn">
//  Authors: Gilles Louppe <g.louppe@gmail.com>
//           Peter Prettenhofer <peter.prettenhofer@gmail.com>
//           Brian Holt <bdholt1@gmail.com>
//           Noel Dawe <noel@dawe.me>
//           Satrajit Gosh <satrajit.ghosh@gmail.com>
//           Lars Buitinck <L.J.Buitinck@uva.nl>
//           Sergey Zyuzin
//  Licence: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Tree
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal abstract class SplitterBase
    {
        protected readonly ICriterion criterion;
        protected uint max_features;
        protected uint min_samples_leaf;
        private readonly Random random_state;
        protected uint start;
        protected uint end;
        protected uint[] samples;
        internal uint n_samples;
        protected uint[] features;
        protected uint n_features;
        private double[] y;
        private uint y_stride;
        private double[] sample_weight;
        protected double[] X;
        protected uint X_stride;
        protected uint rand_r_state;

        public SplitterBase(ICriterion criterion,
                        uint max_features,
                        uint min_samples_leaf,
                        Random random_state)
        {
            this.criterion = criterion;
            this.max_features = max_features;
            this.min_samples_leaf = min_samples_leaf;
            this.random_state = random_state ?? new Random();
        }

        /// <summary>
        /// Initialize the splitter.
        /// </summary>
        /// <param name="?"></param>
        public virtual void init(Matrix<double> X,
                         Matrix<double> y,
                         double[] sample_weight)
        {
            // Reset random state
            //this.rand_r_state = (uint)this.random_state.Next(0, (int)Util.RAND_R_MAX);
            this.rand_r_state = 209652396;

            // Initialize samples and features structures
            int n_samples = X.RowCount;
            uint[] samples = new uint[n_samples];


            uint j = 0;

            for (uint i = 0; i < n_samples; i++)
            {
                // Only work with positively weighted samples
                if (sample_weight == null || sample_weight[i] != 0.0)
                {
                    samples[j] = i;
                    j += 1;
                }
            }

            this.samples = samples;
            this.n_samples = j;

            uint n_features = (uint)X.ColumnCount;
            uint[] features = new uint[n_features];

            for (uint i = 0; i < n_features; i++)
            {
                features[i] = i;
            }


            this.features = features;
            this.n_features = n_features;


            // Initialize X, y, sample_weight
            this.X = X.ToRowWiseArray();
            this.X_stride = (uint)X.ColumnCount;
            this.y = y.ToRowWiseArray();
            this.y_stride = (uint)y.ColumnCount;
            this.sample_weight = sample_weight;
        }

        ///Reset splitter on node samples[start:end].    
        public void node_reset(uint start, uint end, ref double impurity)
        {
            this.start = start;
            this.end = end;


            this.criterion.init(this.y,
                                this.y_stride,
                                this.sample_weight,
                                this.samples,
                                start,
                                end);


            impurity = this.criterion.node_impurity();
        }

        /// <summary>
        /// Find a split on node samples[start:end].
        /// </summary>
        /// <param name="pos"></param>
        /// <param name="feature"></param>
        /// <param name="threshold"></param>
        public abstract void node_split(ref uint pos, ref uint feature, ref double threshold);

        /// <summary>
        /// Copy the value of node samples[start:end] into dest.
        /// </summary>
        /// <param name="dest"></param>
        public void node_value(double[] dest, uint offset)
        {
            this.criterion.node_value(dest, offset);
        }
    }
}
