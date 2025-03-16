import Image from "next/image";
import { Button } from "../ui/button";

type ActiveRideProps = {
  rideSelected: boolean;
  pickupLocation?: string;
  dropoffLocation?: string;
};

const ActiveRide: React.FC<ActiveRideProps> = ({ 
  rideSelected, 
  pickupLocation = "", 
  dropoffLocation = "" 
}) => {
  return (
    <section className="w-full h-[30vh] flex flex-col items-center justify-center p-3 bg-gray-900 text-yellow-400 border-t border-neutral-300 shadow-sm">
      {!rideSelected ? (
        <div className="flex flex-col items-center gap-4 py-8">
          <div className="animate-pulse flex flex-row gap-2">
            <div className="h-2.5 w-2.5 bg-yellow-400 rounded-full"></div>
            <div className="h-2.5 w-2.5 bg-yellow-400 rounded-full"></div>
            <div className="h-2.5 w-2.5 bg-yellow-400 rounded-full"></div>
          </div>
          <h2 className="text-xl text-yellow-400 font-semibold text-gray-800">
            Finding Rides for you...
          </h2>
          <span className="text-sm text-neutral-200">
            Please wait while we connect you with a driver
          </span>
        </div>
      ) : (
        <div className="flex flex-row items-start justify-between w-full pt-3">
          <div className="flex flex-col w-full">
            <div className="flex items-center gap-3 mb-5">
              <div className="relative h-12 w-12 rounded-full overflow-hidden border border-gray-200">
                <Image
                  src="/pj.png"
                  alt="Driver"
                  width={64}
                  height={64}
                  className="object-cover"
                />
              </div>
              <div>
                <h3 className="font-medium text-base text-gray-200">PJ boii</h3>
                <div className="flex items-center text-gray-500 text-xs">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-3.5 w-3.5 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                    />
                  </svg>
                  1234567890
                </div>
              </div>
              <div className="flex flex-col gap-2 items-center ml-auto">
                <Button
                disabled={true}
                  className="w-24 ml-auto text-md tracking-tight font-semibold text-yellow-400 bg-black hover:bg-gray-950 py-1 h-8 rounded-full"
                  variant="ghost"
                >
                  Reached
                </Button>
                <Button 
                className="w-24 ml-auto text-md tracking-tight font-semibold text-white bg-red-600 hover:bg-gray-500 py-1 h-8 rounded-full" 
                variant="ghost"
              >
                Cancel
              </Button>
              </div>
            </div>

            <div className="rounded-xl p-4 w-full">
              <div className="flex flex-col space-y-4">
                <div className="flex items-start">
                  <div className="mr-3 text-xs font-medium text-gray-400 w-16">
                    PICKUP
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-200">
                      {pickupLocation}
                    </p>
                  </div>
                </div>

                <div className="border-t border-gray-200 pt-4 flex items-start">
                  <div className="mr-3 text-xs font-medium text-gray-400 w-16">
                    DROP OFF
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-200">
                      {dropoffLocation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default ActiveRide;