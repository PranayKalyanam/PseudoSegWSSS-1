for feature_map, directory in pseudo_mask_results.output_paths.items():

            self.logger.info(
                f"  {feature_map:<10} : {directory}"
            )

        self.logger