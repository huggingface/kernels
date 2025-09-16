# Kernels CLI Reference

## Main Functions

### kernels upload

Use `kernels upload <dir_containing_build> --repo_id="hub-username/kernel"` to upload
your kernel builds to the Hub. 

**Notes**:

* This will take care of creating a repository on the Hub with the `repo_id` provided. 
* If a repo with the `repo_id` already exists and if it contains a `build` with the build variant
being uploaded, it will attempt to delete the files existing under it.
* Make sure to be authenticated (run `hf auth login` if not) to be able to perform uploads to the Hub.