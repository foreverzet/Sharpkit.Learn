// -----------------------------------------------------------------------
// <copyright file="RegressionCriterion.cs" company="Sharpkit.Learn">
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

    /// <summary>
    /// Abstract criterion for regression.
    ///
    /// Computes variance of the target values left and right of the split point.
    /// Computation is linear in `n_samples` by using ::
    ///
    /// var = \sum_i^n (y_i - y_bar) ** 2
    ///      = (\sum_i^n y_i ** 2) - n_samples y_bar ** 2
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal abstract class RegressionCriterion : ICriterion
    {
        private double[] mean_left;
        private double[] mean_right;
        protected double[] mean_total;
        private double[] sq_sum_left;
        private double[] sq_sum_right;
        protected double[] sq_sum_total;
        protected double[] var_left;
        protected double[] var_right;
        private double[] y;
        private uint y_stride;
        private double[] sample_weight;
        private uint[] samples;
        private uint start;
        private uint end;
        private uint n_node_samples;
        protected uint n_outputs;
        private uint pos;
        private double weighted_n_left;
        private double weighted_n_right;
        protected double weighted_n_node_samples;


        public RegressionCriterion(uint n_outputs)
        {
            // Allocate accumulators
            this.n_outputs = n_outputs;
            this.mean_left = new double[n_outputs];
            this.mean_right = new double[n_outputs];
            this.mean_total = new double[n_outputs];
            this.sq_sum_left = new double[n_outputs];
            this.sq_sum_right = new double[n_outputs];
            this.sq_sum_total = new double[n_outputs];
            this.var_left = new double[n_outputs];
            this.var_right = new double[n_outputs];
        }

        /// <summary>
        /// Initialize the criterion at node samples[start:end] and
        /// children samples[start:start] and samples[start:end].
        /// </summary>
        /// <param name="y"></param>
        /// <param name="y_stride"></param>
        /// <param name="sample_weight"></param>
        /// <param name="samples"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        public void Init(double[] y,
                         uint y_stride,
                         double[] sample_weight,
                         uint[] samples,
                         uint start,
                         uint end)
        {
            // Initialize fields
            this.y = y;
            this.y_stride = y_stride;
            this.sample_weight = sample_weight;
            this.samples = samples;
            this.start = start;
            this.end = end;
            this.n_node_samples = end - start;
            double weighted_n_node_samples = 0.0;


            // Initialize accumulators


            double y_ik = 0.0;
            double w = 1.0;


            Array.Clear(mean_left, 0,(int)n_outputs);
            Array.Clear(mean_right, 0, (int)n_outputs);
            Array.Clear(mean_total, 0, (int)n_outputs);
            Array.Clear(sq_sum_right, 0, (int)n_outputs);
            Array.Clear(sq_sum_left, 0, (int)n_outputs);
            Array.Clear(sq_sum_total, 0, (int)n_outputs);
            Array.Clear(var_left, 0, (int)n_outputs);
            Array.Clear(var_right, 0, (int)n_outputs);

            for (uint p = start; p < end; p++)
            {
                uint i = samples[p];


                if (sample_weight != null)
                {
                    w = sample_weight[i];
                }

                for (int k = 0; k < n_outputs; k++)
                {
                    y_ik = y[i*y_stride + k];
                    sq_sum_total[k] += w*y_ik*y_ik;
                    mean_total[k] += w*y_ik;
                }

                weighted_n_node_samples += w;
            }

            this.weighted_n_node_samples = weighted_n_node_samples;


            for (int k = 0; k < n_outputs; k++)
            {
                mean_total[k] /= weighted_n_node_samples;
            }


            // Reset to pos=start
            this.Reset();
        }

        /// <summary>
        /// Reset the criterion at pos=start.
        /// </summary>
        public void Reset()
        {
            this.pos = this.start;

            this.weighted_n_left = 0.0;
            this.weighted_n_right = this.weighted_n_node_samples;


            for (int k = 0; k < n_outputs; k++)
            {
                mean_right[k] = mean_total[k];
                mean_left[k] = 0.0;
                sq_sum_right[k] = sq_sum_total[k];
                sq_sum_left[k] = 0.0;
                var_left[k] = 0.0;
                var_right[k] = (sq_sum_right[k] -
                                weighted_n_node_samples*(mean_right[k]*
                                                         mean_right[k]));
            }
        }

        /// <summary>
        /// Update the collected statistics by moving samples[pos:new_pos] from
        ///   the right child to the left child.
        /// </summary>
        /// <param name="new_pos"></param>
        public void Update(uint new_pos)
        {
            double w = 1.0;

            // Note: We assume start <= pos < new_pos <= end

            for (uint p = pos; p < new_pos; p++)
            {
                uint i = samples[p];

                if (sample_weight != null)
                {
                    w = sample_weight[i];
                }

                for (int k = 0; k < n_outputs; k++)
                {
                    double y_ik = y[i*y_stride + k];
                    double w_y_ik = w*y_ik;

                    sq_sum_left[k] += w_y_ik*y_ik;
                    sq_sum_right[k] -= w_y_ik*y_ik;


                    mean_left[k] = ((weighted_n_left*mean_left[k] + w_y_ik)/
                                    (weighted_n_left + w));
                    mean_right[k] = ((weighted_n_right*mean_right[k]
                                      - w_y_ik)/
                                     (weighted_n_right - w));
                }

                weighted_n_left += w;
                weighted_n_right -= w;
            }

            for (int k = 0; k < n_outputs; k++)
            {
                var_left[k] = (sq_sum_left[k] -
                               weighted_n_left*(mean_left[k]*mean_left[k]));
                var_right[k] = (sq_sum_right[k] -
                                weighted_n_right*(mean_right[k]*mean_right[k]));
            }

            this.pos = new_pos;
        }

        public abstract double NodeImpurity();


        public abstract double ChildrenImpurity();

        /// <summary>
        /// Compute the node value of samples[start:end] into dest.
        /// </summary>
        public void NodeValue(double[] dest, uint offset)
        {
            Array.Copy(this.mean_total, 0, dest, offset, n_outputs);
        }
    }
}
