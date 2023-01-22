import pandas
import matplotlib.pyplot as plt
import numpy
from phik.report import plot_correlation_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier

data_file = 'data/autos.csv'
encoders = []


def load_data():
    # index, dateCrawled, name, seller, offerType, price, abtest, vehicleType, yearOfRegistration, gearbox, powerPS,
    # model, kilometer, monthOfRegistration, fuelType, brand, notRepairedDamage, dateCreated, nrOfPictures, postalCode,
    # lastSeen
    data = pandas.read_csv(data_file, delimiter=",", parse_dates=['dateCrawled', 'dateCreated', 'lastSeen'])

    return data


def extract_date(data, label: str):
    data[str(label + 'Year')] = pandas.DatetimeIndex(data[label]).year
    data[str(label + 'Month')] = pandas.DatetimeIndex(data[label]).month
    data[str(label + 'Day')] = pandas.DatetimeIndex(data[label]).day

    data = data.drop(label, axis=1)

    return data


def encode_values(data):
    sellerEncoder = LabelEncoder()
    data['seller'] = sellerEncoder.fit_transform(data['seller'])
    encoders.append(sellerEncoder)  # 0

    offerEncoder = LabelEncoder()
    data['offerType'] = offerEncoder.fit_transform(data['offerType'])
    encoders.append(offerEncoder)  # 1

    vehicleTypeEncoder = LabelEncoder()
    data['vehicleType'] = vehicleTypeEncoder.fit_transform(data['vehicleType'])
    encoders.append(vehicleTypeEncoder)  # 2

    gearboxEncoder = LabelEncoder()
    data['gearbox'] = gearboxEncoder.fit_transform(data['gearbox'])
    encoders.append(gearboxEncoder)  # 3

    modelEncoder = LabelEncoder()
    data['model'] = modelEncoder.fit_transform(data['model'])
    encoders.append(modelEncoder)  # 4

    fuelEncoder = LabelEncoder()
    data['fuelType'] = fuelEncoder.fit_transform(data['fuelType'])
    encoders.append(fuelEncoder)  # 5

    brandEncoder = LabelEncoder()
    data['brand'] = brandEncoder.fit_transform(data['brand'])
    encoders.append(brandEncoder)  # 6

    damageEncoder = LabelEncoder()
    data['notRepairedDamage'] = damageEncoder.fit_transform(data['notRepairedDamage'])
    encoders.append(damageEncoder)  # 7

    return data


def calculate_phik(data, title: str):
    phik_corr_matrix = data.phik_matrix()
    plot_correlation_matrix(phik_corr_matrix.values,
                            x_labels=phik_corr_matrix.columns,
                            y_labels=phik_corr_matrix.index,
                            vmin=0, vmax=1, color_map="Greens",
                            title=title,
                            fontsize_factor=1.5,
                            figsize=(16, 14))
    plt.tight_layout()
    plt.show()


def global_correlation(data, title: str):
    global_corr, labels = data.global_phik()
    plot_correlation_matrix(global_corr,
                            x_labels=[''],
                            y_labels=labels,
                            vmin=0, vmax=1,
                            title=title,
                            fontsize_factor=1.5,
                            figsize=(5, 8))
    plt.tight_layout()
    plt.show()


def compare_price_fuelType(data):
    df_fuel = data[['fuelType', 'price']]
    fuel_price = df_fuel.groupby('fuelType').mean()
    plt.figure(figsize=(8, 5))
    plt.bar(encoders[5].inverse_transform(fuel_price.index), fuel_price['price'])
    plt.title("Mean price of used cars by fuel type")
    plt.xlabel("Fuel type")
    plt.ylabel("Price")
    plt.show()


def compare_price_gearbox(data):
    df_gear = data[['gearbox', 'price']]
    gearbox_price = df_gear.groupby('gearbox').mean()
    plt.figure(figsize=(6, 4))
    plt.bar(encoders[3].inverse_transform(gearbox_price.index), gearbox_price['price'])
    plt.title("Mean price of used cars by gearbox type")
    plt.xlabel("Gearbox type")
    plt.ylabel("Price")
    plt.show()


def linear_regression_price(data: pandas.DataFrame):
    X = data.drop('price', axis=1)
    X = X.to_numpy()

    y = data['price']
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred: numpy.ndarray = model.predict(X_test)

    generate_graph(y_pred, y_test, "Linear regression")

    print("Mean absolute error", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print("Mean square error", mean_squared_error(y_true=y_test, y_pred=y_pred))


def generate_graph(y_pred, y_test, title: str):
    df = pandas.DataFrame([['S 1', 'S 1', 'S 2', 'S 2', 'S 3', 'S 3', 'S 4', 'S 4', 'S 5', 'S 5', 'S 6', 'S 6',
                            'S 7', 'S 7', 'S 8', 'S 8'],
                           ['p', 'e', 'p', 'e', 'p', 'e', 'p', 'e', 'p', 'e', 'p', 'e', 'p', 'e', 'p', 'e'],
                           [y_pred[0], y_test[0], y_pred[1], y_test[1],
                            y_pred[2], y_test[2], y_pred[3], y_test[3],
                            y_pred[4], y_test[4], y_pred[5], y_test[5],
                            y_pred[6], y_test[6], y_pred[7], y_test[7]]]).T
    df.columns = ['Sample', 'type', 'Price']
    df.set_index(['Sample', 'type'], inplace=True)
    df.unstack().plot.bar()
    plt.legend(["Expected", "Predicted"])
    plt.title(title)
    plt.show()


def random_forest_price(data: pandas.DataFrame):
    X = data.drop('price', axis=1)
    X = X.to_numpy()

    y = data['price']
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    generate_graph(y_pred, y_test, "Random Forest")

    print("Mean absolute error", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print("Mean square error", mean_squared_error(y_true=y_test, y_pred=y_pred))


# load data
loaded_data = load_data()

# delete unnecessary columns from data
loaded_data = loaded_data.drop(['index', 'dateCrawled', 'abtest', 'nrOfPictures',
                                'lastSeen', 'dateCreated', 'name', 'postalCode'], axis=1)

# delete rows with nan values
loaded_data = loaded_data.dropna()

# delete rows where powerPS is 0
loaded_data = loaded_data[loaded_data['powerPS'] != 0]

# delete rows where price is 0
loaded_data = loaded_data[loaded_data['price'] != 0]

# delete extreme values
loaded_data = loaded_data[loaded_data['price'] < 100000]
loaded_data = loaded_data[loaded_data['price'] > 500]

# calculate phik correlation matrix
# calculate_phik(loaded_data, r"Correlation $\phi_K$")

# encode non-integer values
loaded_data = encode_values(loaded_data)

# calculate global correlation
# global_correlation(loaded_data, 'Global correlation')

# delete offerType attribute
loaded_data = loaded_data.drop(['offerType', 'powerPS', 'seller'], axis=1)

# comparison of interesting attributes
# compare_price_fuelType(loaded_data)
# compare_price_gearbox(loaded_data)

# simple linear regression
# linear_regression_price(loaded_data)

# random forest
# random_forest_price(loaded_data)

print(loaded_data.head().to_string())
