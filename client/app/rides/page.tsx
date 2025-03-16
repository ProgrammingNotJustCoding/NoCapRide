"use client";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";

const Rides = () => {
  const [showSurgeDetails, setShowSurgeDetails] = useState(false);

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
                <span className="text-sm text-gray-400">Avg. Ride Distance</span>
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
                          during surge hours to earn priority access for intercity trips, 
                          rentals, and premium ride services! You&apos;ve completed 3 rides so far.
                        </p>
                      </div>
                      <Button
                        onClick={() => setShowSurgeDetails(!showSurgeDetails)}
                        className="bg-yellow-400 hover:bg-yellow-500 text-gray-900 self-start text-sm py-1 h-8"
                      >
                        {showSurgeDetails ? "Hide Details" : "View Surge Map"}
                      </Button>
                    </div>
                  </div>

                  {showSurgeDetails && (
                    <div className="p-4 bg-gray-800 border-t border-gray-700">
                      <div className="mb-4">
                        <h4 className="text-base font-semibold text-white mb-2">
                          Current Surge Areas
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Complete rides in surge pricing areas to earn your priority pass faster. 
                          Rides in higher surge areas count as multiple rides toward your goal.
                        </p>

                        <div className="bg-gray-900 rounded-lg p-4 h-56 relative">
                          {/* This would be replaced with an actual map component in production */}
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center text-gray-400">
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className="h-10 w-10 mx-auto mb-2 text-gray-600"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                                />
                              </svg>
                              <p>Surge Map Visualization</p>
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
                        <h5 className="text-sm font-semibold text-white mb-2">Priority Pass Benefits:</h5>
                        <ul className="text-xs text-gray-300 space-y-1">
                          <li className="flex items-start">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Priority matching for intercity rides
                          </li>
                          <li className="flex items-start">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Discounted rental booking fees
                          </li>
                          <li className="flex items-start">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Premium ride service access
                          </li>
                          <li className="flex items-start">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Exclusive promotions and discounts
                          </li>
                        </ul>
                      </div>

                      <div className="flex justify-end mt-4">
                        <Button className="bg-transparent hover:bg-gray-700 text-yellow-400 border border-yellow-400 text-xs mr-2">
                          View Forecast
                        </Button>
                        <Button className="bg-yellow-400 hover:bg-yellow-500 text-gray-900 text-xs">
                          Navigate to Highest Surge
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </section>

              {/* Progress toward Priority Pass */}
              <section className="mb-6">
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                  <h3 className="text-base font-semibold text-white mb-3">Your Progress</h3>
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
                    <span className="text-yellow-400 font-semibold">3 rides completed</span> toward 
                    your Priority Pass. Complete 7 more rides during surge hours to unlock benefits.
                  </p>
                  <div className="mt-3 text-xs text-gray-400">
                    <p>Your progress resets in: <span className="text-white">6 days 23 hours</span></p>
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
                      <span className="text-xs text-gray-300">Saved Places</span>
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