FROM tensorflow/serving

# Define metadata
LABEL author="tuenguyen"
LABEL version="1.0"
LABEL description="Deploy tensorflow"

# Copy model
# WORKDIR /models
RUN mkdir -p /models/text/1
EXPOSE 8051
ENTRYPOINT ["tensorflow_model_server", "--model_base_path=/models/text/","--model_name=text_cl"]
CMD ["--rest_api_port=8500","--port=8501"]