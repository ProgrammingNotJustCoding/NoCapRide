const RideRequestOverlay: React.FC<{
  timer: number;
  pickupDistance: number;
  rideDistance: number;
  pickupLocation: string;
  dropoffLocation: string;
  basePrice: number;
  extraAmount: number;
  surgeOptedIn: boolean | null;
  surgeMultiplier: number;
  totalPrice: number;
  onIncreaseExtra: () => void;
  onDecreaseExtra: () => void;
  onAccept: () => void;
  onDecline: () => void;
}> = ({
  timer,
  pickupDistance,
  rideDistance,
  pickupLocation,
  dropoffLocation,
  basePrice,
  extraAmount,
  surgeOptedIn,
  surgeMultiplier,
  totalPrice,
  onIncreaseExtra,
  onDecreaseExtra,
  onAccept,
  onDecline,
}) => {
  const hasSurge = surgeMultiplier > 1.0;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 animate-fadeIn z-[1000]">
      <div className="bg-gray-800 p-6 rounded-xl shadow-2xl max-w-md w-full mx-4 transform scale-100 animate-scaleIn">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-300">New Ride Request</h2>
          <div className="bg-gray-950 px-3 py-1 rounded-full text-yellow-400 font-semibold">
            {timer}s
          </div>
        </div>

        <div className="mb-5">
          <div className="text-gray-200 font-semibold mb-2">
            New ride request at {pickupDistance}km away.
          </div>
          <div className="text-emerald-600 font-bold mb-3">
            {rideDistance}km ride!
          </div>

          <div className="flex items-center mb-2">
            <div className="w-4 h-4 rounded-full bg-green-500 mr-3"></div>
            <span className="text-gray-300">{pickupLocation}</span>
          </div>

          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-red-500 mr-3"></div>
            <span className="text-gray-300">{dropoffLocation}</span>
          </div>
        </div>

        <div className="border-t border-gray-200 pt-4 mb-4">
          <div className="flex justify-between items-center mb-3">
            <span className="text-gray-300">Base Price:</span>
            <span className="font-semibold text-gray-200">&#8377;{basePrice.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center mb-3">
            <span className="text-gray-300">Time Price:</span>
            <span className="font-semibold text-gray-200">&#8377;30</span>
          </div>

          {hasSurge && surgeOptedIn === true && (
            <div className="flex justify-between items-center mb-3">
              <span className="text-gray-300">Surge Multiplier:</span>
              <span className="font-semibold text-yellow-400">
                ({surgeMultiplier}x)
              </span>
            </div>
          )}

          <div className="flex justify-between items-center mb-3">
            <span className="text-gray-300">Add Extra:</span>
            <div className="flex items-center">
              <button
                onClick={onDecreaseExtra}
                className="bg-gray-200 text-gray-800 w-8 h-8 rounded-full flex items-center justify-center mr-2"
                disabled={extraAmount === 0}
              >
                -
              </button>
              <span className="w-10 text-center text-gray-200 font-semibold">
                &#8377;{extraAmount}
              </span>
              <button
                onClick={onIncreaseExtra}
                className="bg-gray-200 text-gray-800 w-8 h-8 rounded-full flex items-center justify-center ml-2"
              >
                +
              </button>
            </div>
          </div>

          <div className="flex justify-between items-center mb-4 text-lg text-gray-300 font-bold">
            <span>Total Price:</span>
            <span>&#8377;{totalPrice.toFixed(2)}</span>
          </div>
        </div>

        <div className="flex space-x-4 justify-center">
          <button
            onClick={onDecline}
            className="border border-gray-800 text-gray-800 bg-white font-bold py-2 px-6 rounded-lg transition-colors hover:bg-gray-100"
          >
            Decline
          </button>
          <button
            onClick={onAccept}
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded-lg transition-colors"
          >
            {extraAmount === 0 ? "Accept" : "Request"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default RideRequestOverlay;