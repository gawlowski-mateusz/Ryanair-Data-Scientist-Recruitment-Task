import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TOWAnalyzer:
    def __init__(self):
        """Initialize TOW Analyzer"""      
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self, csv_file_path):
        """Load CSV data and perform initial exploration"""
        # Load data
        self.df = pd.read_csv(csv_file_path, sep='\t')
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['DepartureDate'].min()} to {self.df['DepartureDate'].max()}")
        
        # Basic info
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== Basic Statistics ===")
        print(self.df.describe())
        
        self.df = self.process_dataframe_features(self.df, is_training=True)
        
    def exploratory_data_analysis(self):
        """Comprehensive EDA with domain-specific insights"""
        print("\n=== Exploratory Data Analysis ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Aircraft TOW Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Actual TOW
        axes[0,0].hist(self.df['ActualTOW'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.df['ActualTOW'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["ActualTOW"].mean():.0f}kg')
        axes[0,0].set_title('Distribution of Actual TOW')
        axes[0,0].set_xlabel('TOW (kg)')
        axes[0,0].legend()
        
        # 2. TOW vs Number of Passengers
        axes[0,1].scatter(self.df['FLownPassengers'], self.df['ActualTOW'], alpha=0.6, color='green')
        axes[0,1].set_title('TOW vs Number of Passengers')
        axes[0,1].set_xlabel('Passengers')
        axes[0,1].set_ylabel('TOW (kg)')
        
        # 3. TOW vs Total Fuel
        axes[0,2].scatter(self.df['ActualTotalFuel'], self.df['ActualTOW'], alpha=0.6, color='orange')
        axes[0,2].set_title('TOW vs Total Fuel')
        axes[0,2].set_xlabel('Fuel (kg)')
        axes[0,2].set_ylabel('TOW (kg)')
        
        # 4. Flight Time vs TOW
        axes[1,0].scatter(self.df['ActualFlightTime'], self.df['ActualTOW'], alpha=0.6, color='brown')
        axes[1,0].set_title('Flight Time vs TOW')
        axes[1,0].set_xlabel('Flight Time (minutes)')
        axes[1,0].set_ylabel('TOW (kg)')
        
        # 5. Fuel per Passenger vs TOW
        axes[1,1].scatter(self.df['FuelPerPassenger'], self.df['ActualTOW'], alpha=0.6, color='pink')
        axes[1,1].set_title('Fuel per Passenger vs TOW')
        axes[1,1].set_xlabel('Fuel per Passenger (kg)')
        axes[1,1].set_ylabel('TOW (kg)')
        
        # 6. Feature Correlation Matrix
        correlation_features = ['ActualTOW', 'FLownPassengers', 'ActualTotalFuel', 'FlightBagsWeight', 
                              'ActualFlightTime', 'EstimatedDistance', 'BagLoadFactor']
        corr_matrix = self.df[correlation_features].corr()
        axes[1,2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1,2].set_xticks(range(len(correlation_features)))
        axes[1,2].set_yticks(range(len(correlation_features)))
        axes[1,2].set_xticklabels(correlation_features, rotation=45)
        axes[1,2].set_yticklabels(correlation_features)
        axes[1,2].set_title('Feature Correlation Matrix')
        
        # Add correlation values
        for i in range(len(correlation_features)):
            for j in range(len(correlation_features)):
                axes[1,2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                             ha='center', va='center', fontsize=8)
        
        # 7. Average TOW by DayOfWeek
        day_tow = self.df.groupby('DayOfWeek')['ActualTOW'].mean()
        axes[2,0].bar(day_tow.index, day_tow.values, color='lightcoral')
        axes[2,0].set_title('Average TOW by DayOfWeek')
        axes[2,0].set_xlabel('DayOfWeek')
        axes[2,0].set_ylabel('Average TOW (kg)')

        for i, (day, value) in enumerate(day_tow.items()):
            axes[2,0].text(i, value + 50, f'{value:.0f}', ha='center', va='bottom', fontsize=8)

        # 8. Average TOW by Route (Top 10)
        route_tow = self.df.groupby('Route')['ActualTOW'].mean().sort_values(ascending=False).head(10)
        axes[2,1].barh(range(len(route_tow)), route_tow.values, color='lightgreen')
        axes[2,1].set_yticks(range(len(route_tow)))
        axes[2,1].set_yticklabels(route_tow.index)
        axes[2,1].set_title('Average TOW by Route (Top 10)')
        axes[2,1].set_xlabel('Average TOW (kg)')

        for i, value in enumerate(route_tow.values):
            axes[2,1].text(value + 50, i, f'{value:.0f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("\n=== Key Correlations with TOW ===")
        tow_correlations = self.df[correlation_features].corr()['ActualTOW'].sort_values(ascending=False)
        for feature, corr in tow_correlations.items():
            if feature != 'ActualTOW':
                print(f"{feature}: {corr:.3f}")
    
    def prepare_features_for_modeling(self):
        """Prepare features for machine learning models"""
        print("\n=== Preparing Features for Modeling ===")
        
        # Features for modeling
        feature_columns = [
            'FLownPassengers', 'ActualTotalFuel', 'FlightBagsWeight', 'BagsCount',
            'ActualFlightTime', 'EstimatedDistance', 'FuelPerPassenger', 
            'BagLoadFactor', 'Month', 'DayOfWeek', 'IsWeekend'
        ]
        
        # Categorical variables encode
        le_airport = LabelEncoder()
        self.df['DepartureAirport_encoded'] = le_airport.fit_transform(self.df['DepartureAirport'])
        self.df['ArrivalAirport_encoded'] = le_airport.fit_transform(self.df['ArrivalAirport'])
        
        feature_columns.extend(['DepartureAirport_encoded', 'ArrivalAirport_encoded'])
        
        # Remove rows with missing target variable
        clean_df = self.df.dropna(subset=['ActualTOW'])
        
        X = clean_df[feature_columns].fillna(clean_df[feature_columns].median())
        y = clean_df['ActualTOW']
        
        print(f"Features selected: {len(feature_columns)}")
        print(f"Samples for modeling: {len(X)}")
        
        return X, y, feature_columns
    
    def train_and_evaluate_models(self, X, y, feature_columns):
        """Train multiple regression models and evaluate performance"""
        print("\n=== Training and Evaluating Models ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_to_test = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_test.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'Regression' in name and 'Random' not in name and 'Gradient' not in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            # Training metrics (to check overfitting)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Train_MAE': train_mae,
                'Train_R²': train_r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"  Test MAE: {mae:.1f} kg")
            print(f"  Test RMSE: {rmse:.1f} kg") 
            print(f"  Test R²: {r2:.3f}")
            print(f"  Test MAPE: {mape:.2f}%")
            print(f"  Train R² vs Test R²: {train_r2:.3f} vs {r2:.3f} (overfitting check)")
        
        self.models = results
    
    def visualize_model_performance(self):
        """Create comprehensive model performance visualizations"""
        print("\n=== Visualizing Model Performance ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        mae_scores = [self.models[name]['MAE'] for name in model_names]
        rmse_scores = [self.models[name]['RMSE'] for name in model_names]
        
        best_model = max(self.models.keys(), key=lambda x: self.models[x]['RMSE'])
        best_results = self.models[best_model]
        
        # 1. Mean Absolute Error by Model
        axes[0,0].bar(model_names, mae_scores, color='lightblue', alpha=0.7)
        axes[0,0].set_title('Mean Absolute Error by Model')
        axes[0,0].set_ylabel('MAE (kg)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        for i, value in enumerate(mae_scores):
            axes[0,0].text(i, value + 1, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. RMSE Score by Model
        axes[0,1].bar(model_names, rmse_scores, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('RMSE Score by Model')
        axes[0,1].set_ylabel('RMSE Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        for i, value in enumerate(rmse_scores):
            axes[0,1].text(i, value + 1, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Predicted vs Actual TOW for best model
        axes[0,2].scatter(best_results['actual'], best_results['predictions'], alpha=0.6)
        axes[0,2].plot([60000, 80000], [60000, 80000], 'r--', label='Perfect Prediction')
        axes[0,2].set_title(f'{best_model}: Predicted vs Actual TOW')
        axes[0,2].set_xlabel('Actual TOW (kg)')
        axes[0,2].set_ylabel('Predicted TOW (kg)')
        axes[0,2].legend()
        
        # 4. Residual Plot for best model
        residuals = best_results['actual'] - best_results['predictions']
        axes[1,0].scatter(best_results['predictions'], residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_title(f'{best_model}: Residual Plot')
        axes[1,0].set_xlabel('Predicted TOW (kg)')
        axes[1,0].set_ylabel('Residuals (kg)')
        
        # 5. Residual Distribution for best model
        axes[1,1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Mean: {residuals.mean():.1f}kg')
        axes[1,1].set_title(f'{best_model}: Residual Distribution')
        axes[1,1].set_xlabel('Residuals (kg)')
        axes[1,1].legend()
        
        # 6. Top 10 Feature Importance for best model
        if hasattr(self.models[best_model]['model'], 'feature_importances_'):
            _, _, feature_columns = self.prepare_features_for_modeling()
            importances = self.models[best_model]['model'].feature_importances_
            feature_importance = sorted(zip(feature_columns, importances), 
                                        key=lambda x: x[1], reverse=True)[:10]
            features, importance_values = zip(*feature_importance)

            axes[1,2].barh(range(len(features)), importance_values, color='purple', alpha=0.7)
            axes[1,2].set_yticks(range(len(features)))
            axes[1,2].set_yticklabels(features)
            axes[1,2].set_title(f'{best_model}: Top 10 Feature Importance')
            axes[1,2].set_xlabel('Importance')

            # Add value labels to horizontal bars
            for i, value in enumerate(importance_values):
                axes[1,2].text(value + 0.005, i, f'{value:.3f}', va='center', fontsize=8)
        else:
            axes[1,2].text(0.5, 0.5, 'Error', 
                        ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Feature Importance')

        
        plt.tight_layout()
        plt.show()
        
        # Print best model summary
        print(f"\n=== Best Model: {best_model} ===")
        print(f"MAE: {best_results['MAE']:.1f} kg")
        print(f"RMSE: {best_results['RMSE']:.1f} kg")
        print(f"R²: {best_results['R²']:.3f}")
        print(f"MAPE: {best_results['MAPE']:.2f}%")
        
    def process_dataframe_features(self, df, is_training=True):
        """Helper method to process features for both training and prediction data"""
        
        # Convert date column
        if 'DepartureDate' in df.columns:
            df['DepartureDate'] = pd.to_datetime(df['DepartureDate'], format='%d/%m/%Y')
        
        # Convert numeric columns
        numeric_columns = ['ActualTotalFuel', 'FLownPassengers', 'FlightBagsWeight', 'BagsCount', 'ActualFlightTime']
        if is_training:
            numeric_columns.append('ActualTOW')
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing bag data (same strategy as training)
        missing_bags = df['BagsCount'].isnull()
        missing_bag_weight = df['FlightBagsWeight'].isnull()
        
        # Estimate missing bag counts (0.7 bags per passenger average)
        df.loc[missing_bags, 'BagsCount'] = (df.loc[missing_bags, 'FLownPassengers'] * 0.7).round()
        
        # Estimate missing bag weights (15kg per bag average)
        df.loc[missing_bag_weight, 'FlightBagsWeight'] = df.loc[missing_bag_weight, 'BagsCount'] * 15
                
        # Fuel efficiency metrics
        df['FuelPerPassenger'] = df['ActualTotalFuel'] / df['FLownPassengers']
        
        # Load factors
        df['BagLoadFactor'] = df['BagsCount'] / df['FLownPassengers']
        
        # Temporal features
        if 'DepartureDate' in df.columns:
            df['Month'] = df['DepartureDate'].dt.month
            df['DayOfWeek'] = df['DepartureDate'].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        route_distances = {
            'KUN-RHO': 2084,  # Kaunas to Rhodes
            'STN-KRK': 1378,  # London Stansted to Krakow
            'KRK-STN': 1378,  # Krakow to London Stansted
            'STN-LCJ': 1313,  # London Stansted to Lodz
            'LBA-TSF': 1346,  # Leeds Bradford to Treviso
            'BCN-CGN': 1132,  # Barcelona to Cologne
            'CGN-BCN': 1132,  # Cologne to Barcelona
            'AGP-BLL': 2354,  # Malaga to Billund
            'NYO-AGP': 2987,  # Stockholm Skavsta to Malaga
            'AGP-SXF': 2112,  # Malaga to Berlin Schönefeld
            'PDL-LIS': 1452,  # Ponta Delgada to Lisbon
            'STN-CIA': 1434,  # London Stansted to Rome Ciampino
            'BRU-PMI': 1329,  # Brussels to Palma de Mallorca
            'BRU-BCN': 1065,  # Brussels to Barcelona
            'MRS-BES': 624,   # Marseille to Brest
            'BES-MRS': 624,   # Brest to Marseille
            'MRS-STN': 1004,  # Marseille to London Stansted
            'STN-MRS': 1004,  # London Stansted to Marseille
            'STN-REU': 1213,  # London Stansted to Reus
            'REU-STN': 1213,  # Reus to London Stansted
            'STN-DUB': 463,   # London Stansted to Dublin
            'DUB-STN': 463,   # Dublin to London Stansted
            'STN-NYO': 1420,  # London Stansted to Stockholm Skavsta
            'GDN-NYO': 789    # Gdansk to Stockholm Skavsta
        }
        mean_distance = sum(route_distances.values()) / len(route_distances)
        df['EstimatedDistance'] = df['Route'].map(route_distances).fillna(mean_distance)
            
        return df
    
    def generate_predictions_for_new_data(self, new_data_path, output_path=None):
        """Generate TOW predictions for new data without ActualTOW column"""
        print(f"\n=== Generating Predictions for New Data ===")
        
        if not self.models:
            raise ValueError("No trained models found. Please train models first using train_and_evaluate_models()")
        
        # Load new data
        new_df = pd.read_csv(new_data_path, sep='\t')
        print(f"New data shape: {new_df.shape}")
        print(f"Columns: {list(new_df.columns)}")
        
        # Create a copy for processing
        processed_df = new_df.copy()
        
        # Apply same preprocessing as training data
        processed_df = self.process_dataframe_features(df=processed_df, is_training=False)
        
        # Encode categorical variables - same encoding as training
        from sklearn.preprocessing import LabelEncoder
        le_dep = LabelEncoder()
        le_arr = LabelEncoder()
        
        # Fit encoders on both training and new data to handle unseen airports
        all_dep_airports = list(set(self.df['DepartureAirport'].unique()) | set(processed_df['DepartureAirport'].unique()))
        all_arr_airports = list(set(self.df['ArrivalAirport'].unique()) | set(processed_df['ArrivalAirport'].unique()))
        
        le_dep.fit(all_dep_airports)
        le_arr.fit(all_arr_airports)
        
        processed_df['DepartureAirport_encoded'] = le_dep.transform(processed_df['DepartureAirport'])
        processed_df['ArrivalAirport_encoded'] = le_arr.transform(processed_df['ArrivalAirport'])
        
        # Select the same features used in training
        feature_columns = [
            'FLownPassengers', 'ActualTotalFuel', 'FlightBagsWeight', 'BagsCount',
            'ActualFlightTime', 'EstimatedDistance', 'FuelPerPassenger', 
            'BagLoadFactor', 'Month', 'DayOfWeek', 'IsWeekend',
            'DepartureAirport_encoded', 'ArrivalAirport_encoded'
        ]
        
        # Prepare features for prediction
        X_new = processed_df[feature_columns].fillna(processed_df[feature_columns].median())
        
        print(f"Features prepared for prediction: {X_new.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # Get best model by lowest RMSE
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['RMSE'])
        best_model = self.models[best_model_name]['model']
        
        print(f"Using {best_model_name} for predictions")
        print(f"Model performance - RMSE: {self.models[best_model_name]['RMSE']:.3f}, MAE: {self.models[best_model_name]['MAE']:.1f} kg")
        
        # Scale features if needed
        if 'Regression' in best_model_name and 'Random' not in best_model_name and 'Gradient' not in best_model_name:
            X_new_scaled = self.scaler.transform(X_new)
            predictions = best_model.predict(X_new_scaled)
        else:
            predictions = best_model.predict(X_new)
        
        # Add predictions to the original dataframe
        new_df['PredictedTOW'] = predictions.round().astype(int)
        
        # Add confidence intervals (approximate)
        mae = self.models[best_model_name]['MAE']
        new_df['PredictionLowerBound'] = (predictions - mae).round().astype(int)
        new_df['PredictionUpperBound'] = (predictions + mae).round().astype(int)
        
        # Display prediction summary
        print(f"\n=== Prediction Summary ===")
        print(f"Predictions generated: {len(predictions)}")
        print(f"Average predicted TOW: {predictions.mean():.0f} kg")
        print(f"TOW range: {predictions.min():.0f} - {predictions.max():.0f} kg")
        print(f"Standard deviation: {predictions.std():.0f} kg")
        
        # Show sample predictions
        print(f"\n=== Sample Predictions ===")
        sample_cols = ['FlightNumber', 'Route', 'FLownPassengers', 'ActualTotalFuel', 'PredictedTOW']
        print(new_df[sample_cols].head(10).to_string(index=False))
        
        # Save results if output path provided
        if output_path:
            new_df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
        
        # Create prediction visualization
        self._visualize_predictions(new_df, best_model_name)
        
        return new_df
    
    def _visualize_predictions(self, df_with_predictions, model_name):
        """Visualize prediction results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        fig.suptitle(f'TOW Predictions using {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Predicted TOW
        axes[0].hist(df_with_predictions['PredictedTOW'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(df_with_predictions['PredictedTOW'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df_with_predictions["PredictedTOW"].mean():.0f}kg')
        axes[0].set_title('Distribution of Predicted TOW')
        axes[0].set_xlabel('Predicted TOW (kg)')
        axes[0].legend()
        
        # 2. Predicted TOW vs Passengers
        axes[1].scatter(df_with_predictions['FLownPassengers'], df_with_predictions['PredictedTOW'], alpha=0.6, color='orange')
        axes[1].set_title('Predicted TOW vs Passengers')
        axes[1].set_xlabel('Number of Passengers')
        axes[1].set_ylabel('Predicted TOW (kg)')
        axes[1].tick_params(axis='x', rotation=90)
        
        # 3. Predicted TOW vs Fuel
        axes[2].scatter(df_with_predictions['ActualTotalFuel'], df_with_predictions['PredictedTOW'], alpha=0.6, color='purple')
        axes[2].set_title('Predicted TOW vs Fuel')
        axes[2].set_xlabel('Total Fuel (kg)')
        axes[2].set_ylabel('Predicted TOW (kg)')
        
        plt.tight_layout()
        plt.show()

# Example usage and analysis workflow
def main():
    """Main analysis workflow"""
    print("=== Aircraft TOW Prediction Analysis ===")
    
    # Initialize analyzer
    analyzer = TOWAnalyzer()
    
    # Load and explore data
    analyzer.load_and_explore_data('training.csv')
    
    # Perform EDA
    analyzer.exploratory_data_analysis()
    
    # Prepare features and train models
    X, y, feature_columns = analyzer.prepare_features_for_modeling()
    analyzer.train_and_evaluate_models(X, y, feature_columns)

    # Generate predictions for new data
    analyzer.generate_predictions_for_new_data(
        new_data_path='validation.csv', 
        output_path='predictions_output.csv'
    )
    
    # Visualize performance
    analyzer.visualize_model_performance()

if __name__ == "__main__":
    main()
