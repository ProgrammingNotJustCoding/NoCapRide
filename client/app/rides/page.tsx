"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface DemandArea {
  ward: string;
  distance: number;
  avg_demand: number;
  active_drivers: number;
  available_drivers: number;
  demand_supply_ratio: number;
  score: number;
}

interface NearbyDemandResponse {
  current_ward: string;
  recommendations: DemandArea[];
  metadata: {
    max_distance: number;
    hours_ahead: number;
    generated_at: string;
    processed_wards: number;
    skipped_wards: number;
  };
}

interface SurgePricing {
  pricing: {
    base_fare: number;
    time_fare: number;
    subtotal: number;
    surge_multiplier: number;
    total_price: number;
  };
  trip_details: {
    distance_km: number;
    duration_min: number;
    region: string;
    pricing_time: string;
  };
  demand_supply: {
    forecast_requests: number;
    active_drivers: number;
    available_drivers: number;
    demand_supply_ratio: number;
  };
  pricing_constants: {
    minimum_price: number;
    per_km_value: number;
    per_min_charge: number;
    alpha: number;
  };
}

interface ForecastData {
  datetime: string;
  forecast_requests: number;
}

interface ForecastResponse {
  forecast: ForecastData[];
  metadata: {
    data_type: string;
    hours: number;
    region: string;
    generated_at: string;
  };
}

const formatPrice = (price: number | undefined) => {
  if (typeof price === "number") {
    return `₹${price.toFixed(2)}`;
  }
  return "₹0.00";
};

const Rides = () => {
  const [showSurgeDetails, setShowSurgeDetails] = useState(false);
  const [currentWard, setCurrentWard] = useState("101");
  const [maxDistance, setMaxDistance] = useState(5);
  const [hoursAhead, setHoursAhead] = useState(3);
  const [loading, setLoading] = useState(false);
  const [demandData, setDemandData] = useState<NearbyDemandResponse | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [distance, setDistance] = useState(5);
  const [duration, setDuration] = useState(20);
  const [surgePricing, setSurgePricing] = useState<SurgePricing | null>(null);
  const [showSurgeCalculator, setShowSurgeCalculator] = useState(false);
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(
    null
  );
  const [showForecast, setShowForecast] = useState(false);
  const [bestTimes, setBestTimes] = useState<
    { hour: number; demand: number }[]
  >([]);
  const [potentialEarnings, setPotentialEarnings] = useState<
    {
      hourRange: string;
      minEarnings: number;
      maxEarnings: number;
      confidence: string;
    }[]
  >([]);

  const todayRides = [
    {
      id: 1,
      time: "08:30 AM",
      from: "123 Main St, Downtown",
      to: "456 Business Ave, Uptown",
      amount: "₹175.50",
      status: "Completed",
    },
    {
      id: 2,
      time: "11:45 AM",
      from: "789 Market St, Westside",
      to: "321 Park Lane, Eastside",
      amount: "₹220.00",
      status: "Completed",
    },
    {
      id: 3,
      time: "03:15 PM",
      from: "567 Tech Park, Southside",
      to: "890 Garden Rd, Northside",
      amount: "₹195.75",
      status: "Completed",
    },
    {
      id: 4,
      time: "06:30 PM",
      from: "432 Mall Circle, Midtown",
      to: "765 Residence Blvd, Suburbs",
      amount: "₹245.25",
      status: "Completed",
    },
  ];

  const fetchDemandData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(
        `http://localhost:8888/api/nearby_high_demand?ward=${
          currentWard || "101"
        }&max_distance=${maxDistance || 5}&hours=${hoursAhead || 3}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch demand data");
      }
      const data = await response.json();
      setDemandData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const calculateSurgePrice = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch("http://localhost:8888/api/surge_pricing", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          type: "ward",
          region: currentWard || "101",
          distance: distance || 5,
          duration: duration || 20,
          alpha: 0.5,
          surge: true,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to calculate surge price");
      }

      const data = await response.json();
      setSurgePricing(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to calculate price"
      );
    } finally {
      setLoading(false);
    }
  };

  const fetchForecastData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(
        `http://localhost:8888/api/forecast?data_type=ward&hours=24&region=${
          currentWard || "101"
        }`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch forecast data");
      }
      const data = await response.json();
      setForecastData(data);

      if (
        data?.forecast &&
        Array.isArray(data.forecast) &&
        data.forecast.length > 0
      ) {
        // Calculate best times to drive
        const hourlyDemand = data.forecast.map((f: ForecastData) => ({
          hour: new Date(f.datetime).getHours(),
          demand: f.forecast_requests || 0,
        }));

        // Sort by demand and get top 3 hours
        const topHours = [...hourlyDemand]
          .sort((a, b) => b.demand - a.demand)
          .slice(0, 3);
        setBestTimes(topHours);

        // Calculate potential earnings for different time slots
        const timeSlots = [
          { start: 6, end: 10, label: "Morning Rush" },
          { start: 12, end: 14, label: "Lunch Hours" },
          { start: 17, end: 21, label: "Evening Rush" },
          { start: 22, end: 5, label: "Night Shift" },
        ];

        const earnings = timeSlots.map((slot) => {
          const slotForecasts = hourlyDemand.filter(
            (h: { hour: number; demand: number }) =>
              slot.start <= h.hour && h.hour <= slot.end
          );

          const avgDemand =
            slotForecasts.length > 0
              ? slotForecasts.reduce(
                  (sum: number, f: { hour: number; demand: number }) =>
                    sum + (f.demand || 0),
                  0
                ) / slotForecasts.length
              : 0;

          const baseEarnings = avgDemand * 150; // Assuming average ride fare of ₹150

          return {
            hourRange: `${slot.label} (${slot.start}:00 - ${slot.end}:00)`,
            minEarnings: Math.round(baseEarnings * 0.8),
            maxEarnings: Math.round(baseEarnings * 1.2),
            confidence:
              avgDemand > 50 ? "High" : avgDemand > 30 ? "Medium" : "Low",
          };
        });

        setPotentialEarnings(earnings);
      } else {
        setBestTimes([]);
        setPotentialEarnings([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch forecast");
      setBestTimes([]);
      setPotentialEarnings([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (showSurgeDetails) {
      fetchDemandData();
    }
  }, [showSurgeDetails, currentWard, maxDistance, hoursAhead]);

  useEffect(() => {
    if (showSurgeCalculator && currentWard) {
      calculateSurgePrice();
    }
  }, [showSurgeCalculator, currentWard, distance, duration]);

  useEffect(() => {
    if (showForecast && currentWard) {
      fetchForecastData();
    }
  }, [showForecast, currentWard]);

  return (
    <div className="h-full overflow-y-auto">
      <main className="bg-gray-900 text-gray-200 pb-20">
        <div className="max-w-5xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-white">Your Rides</h1>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                className="border-gray-700 bg-gray-800 hover:bg-gray-700 text-gray-200"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 mr-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                  />
                </svg>
                Filter
              </Button>
              <Button className="bg-yellow-400 hover:bg-yellow-500 text-gray-900">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 mr-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                  />
                </svg>
                Book Ride
              </Button>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Total Rides</span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  This Week
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">42</h3>
                <span className="text-green-400 text-sm flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-3 w-3 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 10l7-7m0 0l7 7m-7-7v18"
                    />
                  </svg>
                  8.3%
                </span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Rides Needed</span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  For Priority
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">8</h3>
                <span className="text-green-400 text-sm flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-3 w-3 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 10l7-7m0 0l7 7m-7-7v18"
                    />
                  </svg>
                  3 completed
                </span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">
                  Avg. Ride Distance
                </span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  This Week
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">4.8 km</h3>
                <span className="text-gray-400 text-sm flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-3 w-3 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M20 12H4"
                    />
                  </svg>
                  0.5%
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {/* Surge Pricing Program */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Surge Pricing Program
                  </h2>
                  <span className="text-xs px-2 py-1 bg-yellow-400 text-gray-900 rounded-full">
                    Active Now
                  </span>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                  <div className="flex flex-col sm:flex-row">
                    <div className="sm:w-2/5 relative h-40">
                      <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/20 to-transparent z-10"></div>
                      <img
                        src="https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZHJpdmluZ3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60"
                        alt="Surge Hour"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="p-4 sm:w-3/5 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center mb-1">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5 text-yellow-400 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M13 10V3L4 14h7v7l9-11h-7z"
                            />
                          </svg>
                          <h3 className="text-lg font-bold text-white">
                            Priority Pass Program
                          </h3>
                        </div>
                        <p className="text-gray-300 text-sm mb-3">
                          Complete{" "}
                          <span className="font-semibold text-yellow-400">
                            10 rides
                          </span>{" "}
                          during surge hours to earn priority access for
                          intercity trips, rentals, and premium ride services!
                          You&apos;ve completed 3 rides so far.
                        </p>
                      </div>
                      <Button
                        onClick={() => setShowSurgeDetails(!showSurgeDetails)}
                        className="bg-yellow-400 hover:bg-yellow-500 text-gray-900 self-start text-sm py-1 h-8"
                      >
                        {showSurgeDetails
                          ? "Hide Details"
                          : "View Surge Details"}
                      </Button>
                    </div>
                  </div>

                  {showSurgeDetails && (
                    <div className="p-4 bg-gray-800 border-t border-gray-700">
                      <div className="mb-4">
                        <h4 className="text-base font-semibold text-white mb-2">
                          Nearby High Demand Areas
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Complete rides in high demand areas to earn your
                          priority pass faster. Higher demand areas may have
                          surge pricing and count as multiple rides toward your
                          goal.
                        </p>

                        {/* Parameters Form */}
                        <div className="bg-gray-900 rounded-lg p-4 mb-4 border border-gray-700">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-1">
                                Current Ward
                              </label>
                              <Input
                                type="text"
                                value={currentWard}
                                onChange={(e) => setCurrentWard(e.target.value)}
                                className="bg-gray-800 border-gray-700 text-gray-200"
                                placeholder="Enter ward number"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-1">
                                Max Distance: {maxDistance} km
                              </label>
                              <Slider
                                value={[maxDistance]}
                                onValueChange={(value) =>
                                  setMaxDistance(value[0])
                                }
                                min={1}
                                max={10}
                                step={1}
                                className="py-2"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-300 mb-1">
                                Hours Ahead: {hoursAhead}h
                              </label>
                              <Slider
                                value={[hoursAhead]}
                                onValueChange={(value) =>
                                  setHoursAhead(value[0])
                                }
                                min={1}
                                max={6}
                                step={1}
                                className="py-2"
                              />
                            </div>
                          </div>
                          <div className="mt-4 flex justify-end">
                            <Button
                              onClick={fetchDemandData}
                              className="bg-yellow-400 hover:bg-yellow-500 text-gray-900"
                              disabled={loading}
                            >
                              {loading ? "Loading..." : "Update Demand Data"}
                            </Button>
                          </div>
                        </div>

                        {/* Error Message */}
                        {error && (
                          <div className="bg-red-900/20 border border-red-700 text-red-400 p-3 rounded-lg mb-4">
                            {error}
                          </div>
                        )}

                        {/* Demand Area Cards */}
                        {demandData &&
                        demandData.recommendations &&
                        demandData.recommendations.length > 0 ? (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {demandData.recommendations.map((area) => (
                              <div
                                key={area.ward}
                                className="bg-gray-900 rounded-lg p-3 border border-gray-700"
                              >
                                <div className="flex justify-between items-start mb-2">
                                  <div>
                                    <h5 className="text-white font-medium">
                                      Ward {area.ward}
                                    </h5>
                                    <p className="text-xs text-gray-400">
                                      {area.distance === 0
                                        ? "Current Location"
                                        : `${area.distance} km away`}
                                    </p>
                                  </div>
                                  <span className="px-2 py-0.5 bg-yellow-400/20 text-yellow-400 text-xs rounded-full">
                                    {area.demand_supply_ratio.toFixed(2)}x
                                    Demand
                                  </span>
                                </div>
                                <div className="space-y-1">
                                  <div className="flex justify-between text-xs">
                                    <span className="text-gray-400">
                                      Average Demand:
                                    </span>
                                    <span className="text-gray-300">
                                      {area.avg_demand.toFixed(0)} rides/hour
                                    </span>
                                  </div>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-gray-400">
                                      Active Drivers:
                                    </span>
                                    <span className="text-gray-300">
                                      {area.active_drivers} drivers
                                    </span>
                                  </div>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-gray-400">
                                      Available:
                                    </span>
                                    <span className="text-green-400">
                                      {area.available_drivers} drivers
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-center text-gray-400 py-8">
                            No demand data available. Click "Update Demand Data"
                            to fetch information.
                          </div>
                        )}

                        {loading && (
                          <div className="text-center text-gray-400 py-8">
                            Loading demand data...
                          </div>
                        )}

                        {/* Surge Price Calculator */}
                        <div className="mt-8">
                          <div className="flex items-center justify-between mb-4">
                            <h4 className="text-base font-semibold text-white">
                              Surge Price Calculator
                            </h4>
                            <Button
                              onClick={() =>
                                setShowSurgeCalculator(!showSurgeCalculator)
                              }
                              className="text-xs bg-transparent hover:bg-gray-700 text-yellow-400 border border-yellow-400"
                            >
                              {showSurgeCalculator
                                ? "Hide Calculator"
                                : "Show Calculator"}
                            </Button>
                          </div>

                          {showSurgeCalculator && (
                            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                                <div>
                                  <label className="block text-sm font-medium text-gray-300 mb-1">
                                    Distance: {distance} km
                                  </label>
                                  <Slider
                                    value={[distance]}
                                    onValueChange={(value) =>
                                      setDistance(value[0])
                                    }
                                    min={1}
                                    max={50}
                                    step={0.5}
                                    className="py-2"
                                  />
                                </div>
                                <div>
                                  <label className="block text-sm font-medium text-gray-300 mb-1">
                                    Duration: {duration} min
                                  </label>
                                  <Slider
                                    value={[duration]}
                                    onValueChange={(value) =>
                                      setDuration(value[0])
                                    }
                                    min={5}
                                    max={120}
                                    step={5}
                                    className="py-2"
                                  />
                                </div>
                              </div>

                              <Button
                                onClick={calculateSurgePrice}
                                className="w-full bg-yellow-400 hover:bg-yellow-500 text-gray-900 mb-4"
                                disabled={loading}
                              >
                                Calculate Price
                              </Button>

                              {surgePricing && (
                                <div className="space-y-4">
                                  <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-gray-800 p-3 rounded-lg">
                                      <div className="text-xs text-gray-400 mb-1">
                                        Base Fare
                                      </div>
                                      <div className="text-lg font-semibold text-white">
                                        {formatPrice(
                                          surgePricing?.pricing?.base_fare
                                        )}
                                      </div>
                                    </div>
                                    <div className="bg-gray-800 p-3 rounded-lg">
                                      <div className="text-xs text-gray-400 mb-1">
                                        Time Fare
                                      </div>
                                      <div className="text-lg font-semibold text-white">
                                        {formatPrice(
                                          surgePricing?.pricing?.time_fare
                                        )}
                                      </div>
                                    </div>
                                  </div>

                                  <div className="bg-gray-800 p-3 rounded-lg">
                                    <div className="flex justify-between items-center mb-2">
                                      <span className="text-sm text-gray-400">
                                        Subtotal
                                      </span>
                                      <span className="text-sm text-white">
                                        {formatPrice(
                                          surgePricing?.pricing?.subtotal
                                        )}
                                      </span>
                                    </div>
                                    <div className="flex justify-between items-center mb-2">
                                      <span className="text-sm text-gray-400">
                                        Surge Multiplier
                                      </span>
                                      <span className="text-sm text-yellow-400">
                                        {surgePricing?.pricing?.surge_multiplier?.toFixed(
                                          2
                                        ) || "1.00"}
                                        x
                                      </span>
                                    </div>
                                    <div className="flex justify-between items-center pt-2 border-t border-gray-700">
                                      <span className="text-base font-medium text-white">
                                        Total Price
                                      </span>
                                      <span className="text-lg font-semibold text-yellow-400">
                                        {formatPrice(
                                          surgePricing?.pricing?.total_price
                                        )}
                                      </span>
                                    </div>
                                  </div>

                                  <div className="bg-gray-800 p-3 rounded-lg">
                                    <h6 className="text-sm font-medium text-white mb-2">
                                      Current Demand Stats
                                    </h6>
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                      <div>
                                        <span className="text-gray-400">
                                          Forecast Requests:
                                        </span>
                                        <span className="text-white ml-1">
                                          {surgePricing?.demand_supply
                                            ?.forecast_requests || 0}
                                        </span>
                                      </div>
                                      <div>
                                        <span className="text-gray-400">
                                          Active Drivers:
                                        </span>
                                        <span className="text-white ml-1">
                                          {surgePricing?.demand_supply
                                            ?.active_drivers || 0}
                                        </span>
                                      </div>
                                      <div>
                                        <span className="text-gray-400">
                                          Available Drivers:
                                        </span>
                                        <span className="text-green-400 ml-1">
                                          {surgePricing?.demand_supply
                                            ?.available_drivers || 0}
                                        </span>
                                      </div>
                                      <div>
                                        <span className="text-gray-400">
                                          Demand/Supply:
                                        </span>
                                        <span className="text-yellow-400 ml-1">
                                          {surgePricing?.demand_supply?.demand_supply_ratio?.toFixed(
                                            2
                                          ) || "1.00"}
                                        </span>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Driver Insights Section */}
                        <div className="mt-8">
                          <div className="flex items-center justify-between mb-4">
                            <h4 className="text-base font-semibold text-white">
                              Driver Insights
                            </h4>
                            <Button
                              onClick={() => setShowForecast(!showForecast)}
                              className="text-xs bg-transparent hover:bg-gray-700 text-yellow-400 border border-yellow-400"
                            >
                              {showForecast ? "Hide Insights" : "Show Insights"}
                            </Button>
                          </div>

                          {showForecast && (
                            <div className="space-y-6">
                              {/* Best Times to Drive */}
                              <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                                <h5 className="text-sm font-semibold text-white mb-3">
                                  Best Times to Drive Today
                                </h5>
                                <div className="grid grid-cols-3 gap-3">
                                  {bestTimes.map((time, index) => (
                                    <div
                                      key={index}
                                      className="bg-gray-800 p-3 rounded-lg text-center"
                                    >
                                      <div className="text-yellow-400 text-lg font-semibold">
                                        {time.hour}:00
                                      </div>
                                      <div className="text-xs text-gray-400 mt-1">
                                        {time.demand} expected rides
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              {/* Earnings Potential */}
                              <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                                <h5 className="text-sm font-semibold text-white mb-3">
                                  Earnings Potential
                                </h5>
                                <div className="space-y-3">
                                  {potentialEarnings.map((slot, index) => (
                                    <div
                                      key={index}
                                      className="bg-gray-800 p-3 rounded-lg"
                                    >
                                      <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm text-white">
                                          {slot.hourRange}
                                        </span>
                                        <span
                                          className={`text-xs px-2 py-1 rounded-full ${
                                            slot.confidence === "High"
                                              ? "bg-green-900/30 text-green-400 border border-green-700/50"
                                              : slot.confidence === "Medium"
                                              ? "bg-yellow-900/30 text-yellow-400 border border-yellow-700/50"
                                              : "bg-red-900/30 text-red-400 border border-red-700/50"
                                          }`}
                                        >
                                          {slot.confidence} Confidence
                                        </span>
                                      </div>
                                      <div className="text-lg font-semibold text-yellow-400">
                                        ₹{slot.minEarnings} - ₹
                                        {slot.maxEarnings}
                                      </div>
                                      <div className="text-xs text-gray-400 mt-1">
                                        Estimated earnings range
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              {/* Demand Forecast Graph */}
                              {forecastData &&
                              forecastData.forecast &&
                              forecastData.forecast.length > 0 ? (
                                <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                                  <h5 className="text-sm font-semibold text-white mb-3">
                                    24-Hour Demand Forecast
                                  </h5>
                                  <div className="h-64">
                                    <ResponsiveContainer
                                      width="100%"
                                      height="100%"
                                    >
                                      <LineChart
                                        data={forecastData.forecast.map(
                                          (hour) => ({
                                            time:
                                              new Date(
                                                hour.datetime
                                              ).getHours() + ":00",
                                            demand: hour.forecast_requests,
                                            datetime: hour.datetime,
                                          })
                                        )}
                                        margin={{
                                          top: 5,
                                          right: 5,
                                          left: 0,
                                          bottom: 5,
                                        }}
                                      >
                                        <CartesianGrid
                                          strokeDasharray="3 3"
                                          stroke="#374151"
                                        />
                                        <XAxis
                                          dataKey="time"
                                          stroke="#9CA3AF"
                                          tick={{ fill: "#9CA3AF" }}
                                          tickLine={{ stroke: "#4B5563" }}
                                        />
                                        <YAxis
                                          stroke="#9CA3AF"
                                          tick={{ fill: "#9CA3AF" }}
                                          tickLine={{ stroke: "#4B5563" }}
                                          label={{
                                            value: "Expected Rides",
                                            angle: -90,
                                            position: "insideLeft",
                                            fill: "#9CA3AF",
                                          }}
                                        />
                                        <Tooltip
                                          contentStyle={{
                                            backgroundColor: "#1F2937",
                                            border: "1px solid #374151",
                                            borderRadius: "0.375rem",
                                            color: "#E5E7EB",
                                          }}
                                          labelStyle={{ color: "#9CA3AF" }}
                                          formatter={(value) => [
                                            `${value} rides`,
                                            "Expected Demand",
                                          ]}
                                          labelFormatter={(label) =>
                                            `Time: ${label}`
                                          }
                                        />
                                        <Line
                                          type="monotone"
                                          dataKey="demand"
                                          stroke="#FBBF24"
                                          strokeWidth={2}
                                          dot={{
                                            fill: "#FBBF24",
                                            stroke: "#FBBF24",
                                            r: 4,
                                          }}
                                          activeDot={{
                                            fill: "#FBBF24",
                                            stroke: "#F59E0B",
                                            r: 6,
                                            strokeWidth: 2,
                                          }}
                                        />
                                      </LineChart>
                                    </ResponsiveContainer>
                                  </div>
                                  <div className="mt-3 text-xs text-gray-400 text-center">
                                    Hover over points to see detailed demand
                                    predictions
                                  </div>
                                </div>
                              ) : (
                                <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                                  <div className="text-center text-gray-400 py-4">
                                    No forecast data available
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Pricing Legend - Update the existing one */}
                        <div className="mt-8">
                          <h4 className="text-base font-semibold text-white mb-4">
                            Surge Pricing Guide
                          </h4>
                          <div className="grid grid-cols-3 gap-2 mb-2">
                            <div className="bg-gray-900 p-3 rounded-lg text-center">
                              <div className="h-2 w-full bg-gradient-to-r from-green-400 to-yellow-400 rounded mb-2"></div>
                              <div className="text-xs text-gray-400">
                                Base Rate
                              </div>
                              <div className="text-sm text-white font-medium">
                                1.0x - 1.5x
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                Normal demand
                              </div>
                            </div>
                            <div className="bg-gray-900 p-3 rounded-lg text-center">
                              <div className="h-2 w-full bg-gradient-to-r from-yellow-400 to-orange-400 rounded mb-2"></div>
                              <div className="text-xs text-gray-400">
                                Peak Rate
                              </div>
                              <div className="text-sm text-white font-medium">
                                1.5x - 2.0x
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                High demand
                              </div>
                            </div>
                            <div className="bg-gray-900 p-3 rounded-lg text-center">
                              <div className="h-2 w-full bg-gradient-to-r from-orange-400 to-red-400 rounded mb-2"></div>
                              <div className="text-xs text-gray-400">
                                Surge Rate
                              </div>
                              <div className="text-sm text-white font-medium">
                                2.0x - 2.5x
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                Very high demand
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2 mb-2">
                        <div className="bg-gray-900 p-2 rounded-lg text-center">
                          <div className="h-2 w-full bg-gradient-to-r from-yellow-300 to-yellow-400 rounded mb-1"></div>
                          <span className="text-xs text-gray-400">
                            1.2 - 1.5x (1 ride credit)
                          </span>
                        </div>
                        <div className="bg-gray-900 p-2 rounded-lg text-center">
                          <div className="h-2 w-full bg-gradient-to-r from-yellow-400 to-yellow-500 rounded mb-1"></div>
                          <span className="text-xs text-gray-400">
                            1.5 - 2.0x (2 ride credits)
                          </span>
                        </div>
                        <div className="bg-gray-900 p-2 rounded-lg text-center">
                          <div className="h-2 w-full bg-gradient-to-r from-yellow-500 to-amber-600 rounded mb-1"></div>
                          <span className="text-xs text-gray-400">
                            2.0 - 2.5x (3 ride credits)
                          </span>
                        </div>
                      </div>

                      <div className="bg-gray-900 p-3 rounded-lg mt-3">
                        <h5 className="text-sm font-semibold text-white mb-2">
                          Priority Pass Benefits:
                        </h5>
                        <ul className="text-xs text-gray-300 space-y-1">
                          <li className="flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clipRule="evenodd"
                              />
                            </svg>
                            Priority matching for intercity rides
                          </li>
                          <li className="flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clipRule="evenodd"
                              />
                            </svg>
                            Discounted rental booking fees
                          </li>
                          <li className="flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clipRule="evenodd"
                              />
                            </svg>
                            Premium ride service access
                          </li>
                          <li className="flex items-start">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clipRule="evenodd"
                              />
                            </svg>
                            Exclusive promotions and discounts
                          </li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              </section>

              {/* Progress toward Priority Pass */}
              <section className="mb-6">
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                  <h3 className="text-base font-semibold text-white mb-3">
                    Your Progress
                  </h3>
                  <div className="mb-3">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>0 rides</span>
                      <span>10 rides</span>
                    </div>
                    <div className="h-3 bg-gray-700 rounded-full">
                      <div
                        className="h-full bg-gradient-to-r from-yellow-400 to-yellow-500 rounded-full"
                        style={{ width: "30%" }}
                      ></div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-300">
                    <span className="text-yellow-400 font-semibold">
                      3 rides completed
                    </span>{" "}
                    toward your Priority Pass. Complete 7 more rides during
                    surge hours to unlock benefits.
                  </p>
                  <div className="mt-3 text-xs text-gray-400">
                    <p>
                      Your progress resets in:{" "}
                      <span className="text-white">6 days 23 hours</span>
                    </p>
                  </div>
                </div>
              </section>
            </div>

            <div className="lg:col-span-1">
              {/* Rides taken today */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Today&apos;s Rides
                  </h2>
                  <span className="text-sm text-gray-400">
                    {todayRides.length} rides
                  </span>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                  {todayRides.length > 0 ? (
                    <div className="divide-y divide-gray-700 max-h-96 overflow-y-auto">
                      {todayRides.map((ride) => (
                        <div
                          key={ride.id}
                          className="p-3 hover:bg-gray-700 transition-colors"
                        >
                          <div className="flex justify-between mb-2">
                            <div className="flex items-center">
                              <div className="h-8 w-8 rounded-full bg-gray-700 flex items-center justify-center mr-2">
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  className="h-4 w-4 text-yellow-400"
                                  fill="none"
                                  viewBox="0 0 24 24"
                                  stroke="currentColor"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                                  />
                                </svg>
                              </div>
                              <span className="font-medium text-white">
                                {ride.time}
                              </span>
                            </div>
                            <span
                              className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                                ride.status === "Completed"
                                  ? "bg-green-900/30 text-green-400 border border-green-700/50"
                                  : "bg-yellow-900/30 text-yellow-400 border border-yellow-700/50"
                              }`}
                            >
                              {ride.status}
                            </span>
                          </div>
                          <div className="flex items-start mb-2">
                            <div className="mr-2 mt-1">
                              <div className="h-2 w-2 rounded-full bg-green-500"></div>
                              <div className="h-8 w-0.5 bg-gray-600 mx-auto my-1"></div>
                              <div className="h-2 w-2 rounded-full bg-red-500"></div>
                            </div>
                            <div className="flex-1">
                              <div className="mb-2">
                                <p className="text-xs text-gray-400">FROM</p>
                                <p className="text-xs text-gray-300 truncate">
                                  {ride.from}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-400">TO</p>
                                <p className="text-xs text-gray-300 truncate">
                                  {ride.to}
                                </p>
                              </div>
                            </div>
                          </div>
                          <div className="flex justify-between items-center">
                            <div className="text-right font-semibold text-yellow-400">
                              {ride.amount}
                            </div>
                            <button className="text-xs text-gray-400 hover:text-white transition-colors">
                              Details
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-6 text-center text-gray-400">
                      No rides taken today
                    </div>
                  )}
                  <div className="p-3 bg-gray-800 border-t border-gray-700">
                    <Button className="w-full bg-transparent hover:bg-gray-700 text-gray-300 border border-gray-700">
                      View All History
                    </Button>
                  </div>
                </div>
              </section>

              {/* Quick Actions */}
              <section className="mb-6">
                <h2 className="text-lg font-semibold text-white mb-3">
                  Quick Actions
                </h2>
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-3">
                  <div className="grid grid-cols-2 gap-2">
                    <button className="bg-gray-900 hover:bg-gray-700 p-3 rounded-lg flex flex-col items-center justify-center transition-colors border border-gray-700">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 text-yellow-400 mb-1"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      <span className="text-xs text-gray-300">
                        Scheduled Rides
                      </span>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-3 rounded-lg flex flex-col items-center justify-center transition-colors border border-gray-700">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 text-yellow-400 mb-1"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"
                        />
                      </svg>
                      <span className="text-xs text-gray-300">
                        Payment Methods
                      </span>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-3 rounded-lg flex flex-col items-center justify-center transition-colors border border-gray-700">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 text-yellow-400 mb-1"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
                        />
                      </svg>
                      <span className="text-xs text-gray-300">
                        Saved Places
                      </span>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-3 rounded-lg flex flex-col items-center justify-center transition-colors border border-gray-700">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 text-yellow-400 mb-1"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      <span className="text-xs text-gray-300">
                        Help & Support
                      </span>
                    </button>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Rides;
