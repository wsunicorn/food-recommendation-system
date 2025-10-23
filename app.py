from food_recomendation_system.app import app

if __name__ == '__main__':
    # Run the Flask development server
    app.run(host='127.0.0.1', port=5000, debug=True)
