# Target to install dependencies
install:
	pip install flask flask_cors pandas scikit-learn xgboost numpy joblib optuna

train:
	curl -X POST -F "file=@data_before_feb13.csv" http://127.0.0.1:5000/train_model

predict:
	curl -X POST -F "file=@data_before_feb13.csv" http://127.0.0.1:5000/predict_batch


# Help command
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install : Install all required dependencies"
	@echo "  help    : Show this help message"

.PHONY: install train predict help