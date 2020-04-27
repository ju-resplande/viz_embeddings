from subprocess import run

def copy_to_s3(filename, bucket_name):
    """Copy file to AWS S3 bucket.  

    Args:  
        filename (str): file name to copy to S3  
        bucket_name (str): AWS S3 bucket name  
    """
    cmds = ['aws',
            's3',
            'cp',
            filename,
            f's3://{bucket_name}/',
            '--grants',
            'read=uri=http://acs.amazonaws.com/groups/global/AllUsers',
            ]
    run(cmds)
