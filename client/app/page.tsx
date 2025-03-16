"use client";

import FinalLoadoutThingy from "@/components/FinalLoadoutThingy";
import RideRequestOverlay from "@/components/home/RideRequestOverlay";
import SurgeOverlay from "@/components/home/SurgeOverlay";
import ActiveRide from "@/components/home/ActiveRide";
import TopDetails from "@/components/TopDetails";
import { useEffect, useState } from "react";

const Home: React.FC = () => {
  // Ward selection states
  const [wardSelected, setWardSelected] = useState<string>("");
  const [region, setRegion] = useState<string>("");
  const [showWardSelection, setShowWardSelection] = useState(true);

  // Ride states
  const [rideSelected, setRideSelected] = useState(false);
  const [showSurgeOverlay, setShowSurgeOverlay] = useState(false);
  const [showSparkles, setShowSparkles] = useState(false);
  const [surgeOptedIn, setSurgeOptedIn] = useState<boolean>(false);
  const [showRideRequest, setShowRideRequest] = useState(false);
  const [rideRequestTimer, setRideRequestTimer] = useState(60);
  const [extraAmount, setExtraAmount] = useState(0);

  // Ride details
  const [pickupDistance] = useState(1);
  const [rideDistance] = useState(10);
  const [pickupLocation] = useState("123 Main St, Downtown...");
  const [dropoffLocation] = useState("456 Market Ave, Uptown...");
  const [basePrice, setBasePrice] = useState(0);
  const [surgeMultiplier, setSurgeMultiplier] = useState(1);
  const [totalPrice, setTotalPrice] = useState(0);
  const [acceptedRideDetails, setAcceptedRideDetails] = useState({
    pickup: "",
    dropoff: "",
  });
  const [surgeSelected, setSurgeSelected] = useState(false);

  const handleWardSelection = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedWard = e.target.value;
    setWardSelected(selectedWard);
    setRegion(`b_${selectedWard}`);
  };

  const confirmWardSelection = () => {
    if (wardSelected) {
      setShowWardSelection(false);
      checkDemandForecast();
    }
  };

  const checkDemandForecast = async () => {
    try {
      const response = await fetch(
        "http://localhost:8888/api/demand_forecast_ratio",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ward: wardSelected,
            region: region,
            refresh: false,
          }),
        }
      );

      const data = await response.json();

      // Check if hourly ratios is greater than 3 to display surge overlay
      if (data.demand_supply_ratio > 3) {
        setShowSurgeOverlay(true);
      } else {
        // If no surge, show ride request directly
        fetchSurgePricing();
      }
    } catch (error) {
      console.error("Error fetching demand forecast:", error);
      // Fallback to showing ride request
      fetchSurgePricing();
    }
  };

  const fetchSurgePricing = async () => {
    try {
      const now = new Date().toISOString();
      const response = await fetch("http://localhost:8888/api/surge_pricing", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          type: "ward",
          region: region,
          time: now,
          distance: rideDistance,
          surge: surgeOptedIn,
        }),
      });

      const data = await response.json();

      setBasePrice(data.pricing.base_fare);
      setSurgeMultiplier(data.pricing.surge_multiplier);
      setTotalPrice(data.pricing.total_price);

      setTimeout(() => {
        setShowRideRequest(true);
      }, 2000);
    } catch (error) {
      console.error("Error fetching surge pricing:", error);
      // Fallback to default values
      setBasePrice(165);
      setSurgeMultiplier(surgeOptedIn !== false ? 1.2 : 1.0);
      setTotalPrice(165 * (surgeOptedIn !== false ? 1.2 : 1.0));
      setTimeout(() => {
        setShowRideRequest(true);
      }, 2000);
    }
  };

  useEffect(() => {
    if (!showWardSelection && region && surgeSelected) {
      fetchSurgePricing();
    }
  }, [surgeSelected, region, showWardSelection]);

  const handleSurgeOptIn = () => {
    setSurgeOptedIn(true);
    setShowSurgeOverlay(false);
    setShowSparkles(true);
    setSurgeSelected(true);
    console.log("User opted into surge pricing");
  };

  const handleSurgeCancel = () => {
    setSurgeOptedIn(false);
    setShowSurgeOverlay(false);
    setSurgeSelected(true);
    console.log("User declined surge pricing");
  };

  const handleIncreaseExtra = () => {
    setExtraAmount((prev) => prev + 5);
  };

  const handleDecreaseExtra = () => {
    if (extraAmount >= 5) {
      setExtraAmount((prev) => prev - 5);
    }
  };

  const handleAcceptRide = () => {
    setShowRideRequest(false);
    setRideSelected(true);
    setAcceptedRideDetails({
      pickup: pickupLocation,
      dropoff: dropoffLocation,
    });
    console.log("Ride accepted with extra amount:", extraAmount);
  };

  const handleDeclineRide = () => {
    setShowRideRequest(false);
    setRideSelected(false);
    console.log("Ride declined");
  };

  useEffect(() => {
    if (!showRideRequest) return;

    if (rideRequestTimer <= 0) {
      setShowRideRequest(false);
      return;
    }

    const timer = setTimeout(() => {
      setRideRequestTimer((prev) => prev - 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [rideRequestTimer, showRideRequest]);

  return (
    <main className="w-screen h-screen flex flex-col bg-gray-900">
      {showWardSelection ? (
        <div className="fixed inset-0 flex items-center justify-center bg-gray-900 z-[1000]">
          <div className="bg-gray-800 p-8 rounded-xl shadow-2xl max-w-md w-full mx-4">
            <h2 className="text-2xl font-bold text-gray-300 mb-6 text-center">
              Select Ward
            </h2>
            <select
              value={wardSelected}
              onChange={handleWardSelection}
              className="w-full p-3 mb-6 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select a ward</option>
              <option value="101">Ward 101</option>
              <option value="102">Ward 102</option>
              <option value="103">Ward 103</option>
              <option value="150">Ward 150</option>
            </select>

            <button
              onClick={confirmWardSelection}
              disabled={!wardSelected}
              className={`w-full py-3 rounded-lg font-bold transition-colors ${
                wardSelected
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : "bg-gray-600 text-gray-400 cursor-not-allowed"
              }`}
            >
              Confirm
            </button>
          </div>
        </div>
      ) : (
        <>
          <TopDetails showSparkles={showSparkles} />
          <FinalLoadoutThingy />
          <ActiveRide
            rideSelected={rideSelected}
            pickupLocation={acceptedRideDetails.pickup}
            dropoffLocation={acceptedRideDetails.dropoff}
          />

          {showSurgeOverlay && (
            <SurgeOverlay
              onOptIn={handleSurgeOptIn}
              onCancel={handleSurgeCancel}
            />
          )}

          {showRideRequest && (
            <RideRequestOverlay
              timer={rideRequestTimer}
              pickupDistance={pickupDistance}
              rideDistance={rideDistance}
              pickupLocation={pickupLocation}
              dropoffLocation={dropoffLocation}
              basePrice={basePrice}
              extraAmount={extraAmount}
              surgeOptedIn={surgeOptedIn}
              surgeMultiplier={surgeMultiplier}
              totalPrice={totalPrice + extraAmount}
              onIncreaseExtra={handleIncreaseExtra}
              onDecreaseExtra={handleDecreaseExtra}
              onAccept={handleAcceptRide}
              onDecline={handleDeclineRide}
            />
          )}
        </>
      )}
    </main>
  );
};

export default Home;
