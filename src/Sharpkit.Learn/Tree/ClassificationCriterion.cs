// -----------------------------------------------------------------------
// <copyright file="ClassificationCriterion.cs" company="Sharpkit.Learn">
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
    using System.Linq;

    /// <summary>
    /// Abstract criterion for classification.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal abstract class ClassificationCriterion : ICriterion
    {
        protected uint[] n_classes;
        protected uint label_count_stride;
        protected double[] label_count_left;
        protected double[] label_count_right;
        protected double[] label_count_total;
        private double[] y;
        private uint y_stride;
        private double[] sample_weight;
        private uint[] samples;
        private uint start;
        protected double weighted_n_node_samples;
        protected uint n_outputs;
        private uint pos;
        protected double weighted_n_left;
        protected double weighted_n_right;

        public ClassificationCriterion(uint n_outputs, uint[] n_classes)
        {
            // Default values
            this.n_outputs = n_outputs;

            //Count labels for each output
            this.n_classes = new uint[n_outputs];

            Array.Copy(n_classes, this.n_classes, n_outputs);
            this.label_count_stride = n_classes.Max();

            // Allocate counters
            this.label_count_left = new double[n_outputs * label_count_stride];
            this.label_count_right = new double[n_outputs * label_count_stride];
            this.label_count_total = new double[n_outputs * label_count_stride];
        }
        
        public void init(double[] y,
                         uint y_stride,
                         double[] sample_weight,
                         uint[] samples,
                         uint start,
                         uint end)
        {
            // Initialize the criterion at node samples[start:end] and
            // children samples[start:start] and samples[start:end]."""
            // # Initialize fields
            this.y = y;
            this.y_stride = y_stride;
            this.sample_weight = sample_weight;
            this.samples = samples;
            this.start = start;

            double weighted_n_node_samples = 0.0;

            uint offset = 0;
            for (int k = 0; k < n_outputs; k++)
            {
                Array.Clear(label_count_total, (int)offset, (int)n_classes[k]);
                offset += label_count_stride;
            }

            double w = 1.0;
            for (uint p = start; p < end; p++)
            {
                uint i = samples[p];

                if (sample_weight != null)
                {
                    w = sample_weight[i];
                }

                for (uint k = 0; k < n_outputs; k++)
                {
                    uint c = (uint)y[i * y_stride + k];
                    label_count_total[k * label_count_stride + c] += w;
                }

                weighted_n_node_samples += w;
            }

            this.weighted_n_node_samples = weighted_n_node_samples;

            // Reset to pos=start
            this.reset();
        }

        /// <summary>
        /// Reset the criterion at pos=start.
        /// </summary>
        public void reset()
        {
            this.pos = this.start;

            this.weighted_n_left = 0.0;
            this.weighted_n_right = this.weighted_n_node_samples;

            uint label_count_total_offset = 0;
            uint label_count_left_offset = 0;
            uint label_count_right_offset = 0;

            for (int k = 0; k < n_outputs; k++)
            {
                Array.Clear(label_count_left, (int)label_count_left_offset, (int)n_classes[k]);
                Array.Copy(label_count_total, label_count_total_offset, label_count_right, label_count_right_offset,
                           n_classes[k]);

                label_count_total_offset += label_count_stride;
                label_count_left_offset += label_count_stride;
                label_count_right_offset += label_count_stride;
            }
        }

        /// <summary>
        /// Update the collected statistics by moving samples[pos:new_pos] from
        /// the right child to the left child.
        /// </summary>
        /// <param name="new_pos"></param>
        public void update(uint new_pos)
        {
            // Note: We assume start <= pos < new_pos <= end

            double w = 1.0;
            for (uint p = pos; p < new_pos; p++)
            {
                uint i = samples[p];

                if (sample_weight != null)
                {
                    w = sample_weight[i];
                }

                for (uint k = 0; k < n_outputs; k++)
                {
                    uint label_index = (k * label_count_stride + (uint)y[i * y_stride + k]);
                    label_count_left[label_index] += w;
                    label_count_right[label_index] -= w;
                }


                weighted_n_left += w;
                weighted_n_right -= w;
            }

            this.pos = new_pos;
        }

        public abstract double node_impurity();

        public abstract double children_impurity();

        /// <summary>
        /// Compute the node value of samples[start:end] into dest.
        /// </summary>
        /// <param name="dest"></param>
        /// <returns></returns>
        public void node_value(double[] dest, uint offset)
        {
            uint label_count_stride = 0;
            uint label_count_total_offset = 0;
            uint dest_offset = offset;

            for (int k = 0; k < n_outputs; k++)
            {
                Array.Copy(label_count_total, label_count_total_offset, dest, dest_offset, n_classes[k]);
                dest_offset += label_count_stride;
                label_count_total_offset += label_count_stride;
            }
        }
    }
}
