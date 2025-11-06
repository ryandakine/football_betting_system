FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y gcc gcc-c++ cmake3 && \
    ln -s /usr/bin/cmake3 /usr/bin/cmake && \
    yum clean all

# Install dependencies with compatible versions
RUN pip install --no-cache-dir \
    boto3 \
    'numpy<2.0' \
    'pandas<2.1' \
    scikit-learn \
    xgboost \
    aiohttp \
    requests \
    tenacity

# Copy function code
COPY nfl_lambda_handler.py ${LAMBDA_TASK_ROOT}/lambda_function.py
COPY nfl_live_data_fetcher.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]