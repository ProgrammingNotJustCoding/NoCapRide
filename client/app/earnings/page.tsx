"use client";
import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts";
import { Button } from "@/components/ui/button";

const Earnings = () => {
  const weeklyEarningsData = [
    { day: "Mon", earnings: 420 },
    { day: "Tue", earnings: 590 },
    { day: "Wed", earnings: 310 },
    { day: "Thu", earnings: 680 },
    { day: "Fri", earnings: 520 },
    { day: "Sat", earnings: 890 },
    { day: "Sun", earnings: 760 },
  ];

  const recentTransactions = [
    {
      id: 1,
      date: "21 Jul 2023",
      description: "Ride Earnings",
      amount: "₹175.50",
      status: "Completed",
    },
    {
      id: 2,
      date: "20 Jul 2023",
      description: "Surge Bonus",
      amount: "₹80.00",
      status: "Completed",
    },
    {
      id: 3,
      date: "20 Jul 2023",
      description: "Ride Earnings",
      amount: "₹220.00",
      status: "Completed",
    },
    {
      id: 4,
      date: "19 Jul 2023",
      description: "Weekly Incentive",
      amount: "₹350.75",
      status: "Completed",
    },
    {
      id: 5,
      date: "18 Jul 2023",
      description: "Ride Earnings",
      amount: "₹195.25",
      status: "Completed",
    },
  ];

  const earningsBreakdown = [
    { name: "Base Fare", value: 60 },
    { name: "Surge Pricing", value: 25 },
    { name: "Incentives", value: 15 },
  ];

  const COLORS = ['#FBBF24', '#F97316', '#EC4899'];

  return (
    <div className="h-full overflow-y-auto">
      <main className="bg-gray-900 text-gray-200 pb-20">
        <div className="max-w-5xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-white">Your Earnings</h1>
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
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Export Data
              </Button>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Total Earnings</span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  This Week
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">₹4,170</h3>
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
                  12.3%
                </span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Surge Earnings</span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  This Week
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">₹980</h3>
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
                  18.7%
                </span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Avg. Per Ride</span>
                <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-yellow-400">
                  This Week
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <h3 className="text-2xl font-bold text-white">₹185.50</h3>
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
                  5.2%
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {/* Weekly earnings chart */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Earnings Overview
                  </h2>
                  <div className="flex space-x-2">
                    <button className="text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-full border border-gray-700 text-gray-300">
                      Week
                    </button>
                    <button className="text-xs bg-gray-700 hover:bg-gray-700 px-3 py-1 rounded-full border border-gray-700 text-yellow-400">
                      Month
                    </button>
                    <button className="text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-full border border-gray-700 text-gray-300">
                      Year
                    </button>
                  </div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        width={500}
                        height={300}
                        data={weeklyEarningsData}
                        margin={{
                          top: 5,
                          right: 20,
                          left: 0,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#374151"
                          opacity={0.2}
                        />
                        <XAxis dataKey="day" stroke="#9CA3AF" />
                        <YAxis stroke="#9CA3AF" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#1F2937",
                            border: "1px solid #374151",
                            borderRadius: "0.375rem",
                            color: "#E5E7EB",
                          }}
                          formatter={(value) => [`₹${value}`, "Earnings"]}
                        />
                        <Bar
                          dataKey="earnings"
                          fill="#FBBF24"
                          barSize={30}
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>

              {/* Earnings trend */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Earnings Trends
                  </h2>
                  <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-green-400">
                    +15% vs Last Month
                  </span>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={[
                          { month: "Jan", earnings: 2800 },
                          { month: "Feb", earnings: 3200 },
                          { month: "Mar", earnings: 2900 },
                          { month: "Apr", earnings: 3500 },
                          { month: "May", earnings: 3800 },
                          { month: "Jun", earnings: 3600 },
                          { month: "Jul", earnings: 4200 },
                        ]}
                        margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
                        <XAxis dataKey="month" stroke="#9CA3AF" />
                        <YAxis stroke="#9CA3AF" />
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: "#1F2937",
                            border: "1px solid #374151",
                            borderRadius: "0.375rem",
                            color: "#E5E7EB",
                          }}
                          formatter={(value) => [`₹${value}`, "Earnings"]}
                        />
                        <Line
                          type="monotone"
                          dataKey="earnings"
                          stroke="#FBBF24"
                          strokeWidth={2}
                          dot={{ r: 4, strokeWidth: 2 }}
                          activeDot={{ r: 6 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4 grid grid-cols-3 gap-2">
                    <div className="bg-gray-900 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Best Day</div>
                      <div className="flex justify-between items-center">
                        <div className="text-sm font-medium text-white">Sunday</div>
                        <div className="text-yellow-400 font-medium">₹890</div>
                      </div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Best Time</div>
                      <div className="flex justify-between items-center">
                        <div className="text-sm font-medium text-white">6-9 PM</div>
                        <div className="text-yellow-400 font-medium">+35%</div>
                      </div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Monthly Trend</div>
                      <div className="flex justify-between items-center">
                        <div className="text-sm font-medium text-white">July</div>
                        <div className="text-green-400 font-medium">↑ 15%</div>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </div>

            <div className="lg:col-span-1">
              {/* Earnings breakdown */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Earnings Breakdown
                  </h2>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={earningsBreakdown}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          fill="#8884d8"
                          paddingAngle={5}
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          labelLine={false}
                        >
                          {earningsBreakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#1F2937",
                            border: "1px solid #374151",
                            borderRadius: "0.375rem",
                            color: "#E5E7EB",
                          }}
                          formatter={(value) => [`${value}%`, ""]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4">
                    <div className="flex flex-col space-y-2">
                      {earningsBreakdown.map((item, index) => (
                        <div key={item.name} className="flex items-center justify-between">
                          <div className="flex items-center">
                            <div
                              className="h-3 w-3 rounded-full mr-2"
                              style={{ backgroundColor: COLORS[index % COLORS.length] }}
                            ></div>
                            <span className="text-sm text-gray-300">{item.name}</span>
                          </div>
                          <span className="text-sm font-medium text-white">{item.value}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>

              {/* Recent transactions */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Recent Transactions
                  </h2>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                  <div className="divide-y divide-gray-700 max-h-96 overflow-y-auto">
                    {recentTransactions.map((transaction) => (
                      <div
                        key={transaction.id}
                        className="p-3 hover:bg-gray-700 transition-colors"
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium text-white mb-1">
                              {transaction.description}
                            </div>
                            <div className="text-xs text-gray-400">
                              {transaction.date}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-semibold text-yellow-400">
                              {transaction.amount}
                            </div>
                            <div className="text-xs text-green-400">
                              {transaction.status}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="p-3 bg-gray-800 border-t border-gray-700">
                    <Button className="w-full bg-transparent hover:bg-gray-700 text-gray-300 border border-gray-700">
                      View All Transactions
                    </Button>
                  </div>
                </div>
              </section>

              {/* Payout options */}
              <section className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white">
                    Quick Payout
                  </h2>
                </div>
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                  <div className="mb-4">
                    <div className="text-sm text-gray-400 mb-1">Available for payout</div>
                    <div className="text-2xl font-bold text-white">₹3,245.50</div>
                  </div>
                  <Button className="w-full bg-yellow-400 hover:bg-yellow-500 text-gray-900 mb-2">
                    Withdraw to Bank
                  </Button>
                  <div className="text-xs text-center text-gray-400">
                    Usually processed within 24 hours
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

export default Earnings;