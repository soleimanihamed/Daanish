# services/eda/correlation_service/correlation_service.py

from flask import Flask, request, jsonify
import pandas as pd
from utils.eda.correlation import CorrelationAnalyzer

app = Flask(__name__)


@app.route('/correlation', methods=['POST'])
def correlation_service():
    try:
        # Get JSON body
        params = request.json

        # Extract the data and parameters
        data = params.get('data')
        if data is None:
            return jsonify({"error": "Missing 'data' field in the request."}), 400

        num_method = params.get('num_method', 'pearson')
        cat_method = params.get('cat_method', 'cramers_v')
        cat_num_method = params.get('cat_num_method', 'correlation_ratio')
        return_matrix = params.get('return_matrix', False)

        # Convert data to DataFrame
        main_df = pd.DataFrame(data)

        # Perform correlation analysis
        analyzer = CorrelationAnalyzer(main_df)
        corr_df, corr_matrix = analyzer.correlation_matrix(
            num_method=num_method,
            cat_method=cat_method,
            cat_num_method=cat_num_method,
            return_matrix=True
        )

        # Convert results to JSON serializable format
        response = {
            "unified_table": corr_df.to_dict(orient="records")
        }

        if return_matrix:
            response["correlation_matrix"] = corr_matrix.fillna(None).to_dict()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
